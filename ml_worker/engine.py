import asyncio
import json
import os
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import cv2
import numpy as np
import torch
from pgvector.asyncpg import register_vector
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO
from ultralytics.engine.results import Results

warnings.filterwarnings("ignore")


@dataclass
class IndexerConfig:
    frame_skip: int = 15
    yolo_batch_size: int = 32
    florence_batch_size: int = 6
    motion_threshold: int = 1000
    yolo_conf: float = 0.25
    db_dsn: str = "postgresql://postgres:пароль@localhost:5432/sceneseek_test"
    debug_mode: bool = False
    debug_dir: str = "debug_output"


class VideoSearchEngine:
    def __init__(self, config: Optional[IndexerConfig] = None, use_float16: bool = True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_float16 = use_float16 and self.device == "cuda"
        self.config = config if config else IndexerConfig()

        self.pool: Optional[asyncpg.Pool] = None

        print(f"🚀 [Init] Запуск движка на {self.device}...")
        self._load_models()

    async def initialize_db(self) -> None:
        """
        Асинхронная инициализация пула соединений, расширений и схемы БД.
        """
        try:
            # Создаем пул соединений
            self.pool = await asyncpg.create_pool(dsn=self.config.db_dsn)

            if self.pool is None:
                raise ConnectionError("Не удалось создать пул соединений.")

            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                await register_vector(conn)

                ddl_script = """
                    -- 1. Пользователи
                    CREATE TABLE IF NOT EXISTS users (
                        user_id SERIAL PRIMARY KEY,
                        username VARCHAR(100) NOT NULL,
                        role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'scientist', 'analyst'))
                    );

                    -- Добавляем дефолтного юзера, чтобы foreign keys работали сразу
                    INSERT INTO users (user_id, username, role)
                    VALUES (1, 'admin', 'admin')
                    ON CONFLICT (user_id) DO NOTHING;

                    -- 2. Видео
                    CREATE TABLE IF NOT EXISTS videos (
                        video_id SERIAL PRIMARY KEY,
                        uploaded_by_user_id INT REFERENCES users(user_id) ON DELETE SET NULL,
                        title VARCHAR(255) NOT NULL,
                        path VARCHAR(512) NOT NULL,
                        duration FLOAT,
                        fps FLOAT,
                        resolution VARCHAR(20),
                        processing_status VARCHAR(20) DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- 3. События
                    CREATE TABLE IF NOT EXISTS video_events (
                        event_id BIGSERIAL PRIMARY KEY,
                        video_id INT REFERENCES videos(video_id) ON DELETE CASCADE,
                        timestamp FLOAT NOT NULL,
                        caption TEXT NOT NULL,
                        yolo_metadata JSONB DEFAULT '{}'::jsonb,
                        embedding vector(384),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- 4. История поиска
                    CREATE TABLE IF NOT EXISTS search_history (
                        query_id BIGSERIAL PRIMARY KEY,
                        user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
                        query_text TEXT NOT NULL,
                        query_embedding vector(384),
                        search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- 5. Результаты поиска
                    CREATE TABLE IF NOT EXISTS search_results (
                        result_id BIGSERIAL PRIMARY KEY,
                        query_id BIGINT REFERENCES search_history(query_id) ON DELETE CASCADE,
                        found_event_id BIGINT REFERENCES video_events(event_id) ON DELETE CASCADE,
                        similarity_score FLOAT,
                        is_relevant BOOLEAN DEFAULT NULL
                    );

                    -- --- ИНДЕКСЫ ---

                    -- HNSW индекс для векторов (самый важный)
                    CREATE INDEX IF NOT EXISTS idx_events_embedding
                    ON video_events USING hnsw (embedding vector_cosine_ops);

                    -- GIN индекс для JSONB (метаданные YOLO)
                    CREATE INDEX IF NOT EXISTS idx_events_yolo
                    ON video_events USING GIN (yolo_metadata);

                    -- B-Tree индекс для поиска по видео и времени
                    CREATE INDEX IF NOT EXISTS idx_events_video_id
                    ON video_events(video_id, timestamp);
                    """

                # Выполняем весь скрипт создания таблиц
                await conn.execute(ddl_script)

            print(" 📦 [DB] Подключение и миграции успешны (asyncpg).")

        except Exception as e:
            print(f"❌ [DB] Ошибка инициализации: {e}")
            raise e

    async def close(self) -> None:
        """Закрытие пула соединений"""
        if self.pool:
            await self.pool.close()

    def _load_models(self) -> None:
        print(" ├─ [1/3] YOLOv8...")
        self.yolo = YOLO("yolov8n.pt")

        print(" ├─ [2/3] Florence-2...")
        model_id = "microsoft/Florence-2-base-ft"
        dtype = torch.float16 if self.use_float16 else torch.float32

        self.fl_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(self.device).eval()

        self.fl_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        print(" ├─ [3/3] Embedder...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        print(" └─ Готово.")

    async def search(self, query: str, top_k: int = 5, min_score: float = 0.25) -> List[Dict[str, Any]]:
        """
        Асинхронный семантический поиск.
        """
        if not self.pool:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        # 1. Энкодинг запроса
        query_vec: List[float] = self.embedder.encode(query).tolist()

        # 2. SQL запрос
        sql = """
            SELECT
                v.video_id,
                v.title,
                e.timestamp,
                e.caption,
                1 - (e.embedding <=> $1) as score,
                e.yolo_metadata
            FROM video_events e
            JOIN videos v ON e.video_id = v.video_id
            ORDER BY e.embedding <=> $1
            LIMIT $2;
        """

        results = []
        try:
            async with self.pool.acquire() as conn:
                # Регистрируем вектор для текущего соединения, чтобы передать query_vec корректно
                await register_vector(conn)
                rows = await conn.fetch(sql, query_vec, top_k)

                for r in rows:
                    if r["score"] < min_score:
                        continue

                    # Обработка метаданных (asyncpg может вернуть строку или уже dict)
                    meta = r["yolo_metadata"]
                    if isinstance(meta, str):
                        meta = json.loads(meta)

                    results.append(
                        {"video_id": r["video_id"], "video_title": r["title"], "timestamp": round(r["timestamp"], 2), "caption": r["caption"], "score": round(float(r["score"]), 4), "metadata": meta}
                    )
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")

        return results

    async def run_indexing(self, video_path: str, user_id: int = 1) -> None:
        """
        Главный пайплайн индексации.
        """
        if self.config.debug_mode:
            self._setup_debug()

        if not self.pool:
            await self.initialize_db()

        # 1. Регистрируем видео
        video_id = await self._create_video_entry(video_path, user_id)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # MOG2 (Background Subtractor)
        back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

        yolo_buffer: List[Tuple[np.ndarray, float]] = []
        florence_queue: List[Tuple[Image.Image, float, Dict[str, int]]] = []

        # State: {'counts': dict, 'centers': list}
        prev_yolo_state: Optional[Dict[str, Any]] = None

        frame_idx = 0
        stats = {"motion_skip": 0, "yolo_skip": 0, "indexed": 0}

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                # Motion check
                is_motion = False
                if frame_idx % 5 == 0:
                    is_motion = self._check_motion_mog2(back_sub, frame, self.config.motion_threshold)

                if frame_idx % self.config.frame_skip == 0:
                    timestamp = frame_idx / fps

                    if not is_motion:
                        stats["motion_skip"] += 1
                        frame_idx += 1
                        continue

                    yolo_buffer.append((frame, timestamp))

                    # YOLO Batch Processing
                    if len(yolo_buffer) >= self.config.yolo_batch_size:
                        prev_yolo_state = self._process_yolo_batch(yolo_buffer, florence_queue, prev_yolo_state, stats)
                        yolo_buffer = []
                        print(f" ⏳ Indexing: {timestamp:.1f}s / {duration:.1f}s", end="\r")

                    # Florence Batch Processing & Saving
                    if len(florence_queue) >= self.config.florence_batch_size:
                        count = await self._process_florence_batch_and_save(florence_queue, video_id)
                        stats["indexed"] += count
                        florence_queue = []

                frame_idx += 1

            # Обработка остаточных буферов
            if yolo_buffer:
                prev_yolo_state = self._process_yolo_batch(yolo_buffer, florence_queue, prev_yolo_state, stats)
            if florence_queue:
                count = await self._process_florence_batch_and_save(florence_queue, video_id)
                stats["indexed"] += count

            await self._finalize_video_status(video_id, "ready")
            print(f"\n✅ [Done] Индексация завершена. Сохранено {stats['indexed']} событий.")

        except Exception as e:
            print(f"\n❌ Ошибка индексации: {e}")
            await self._finalize_video_status(video_id, "failed")
            import traceback

            traceback.print_exc()
        finally:
            cap.release()

    # --- DB METHODS (ASYNC) ---

    async def _create_video_entry(self, video_path: str, user_id: int) -> int:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        filename = os.path.basename(video_path)
        resolution = f"{w}x{h}"

        sql = """
            INSERT INTO videos (uploaded_by_user_id, title, path, duration, fps, resolution, processing_status)
            VALUES ($1, $2, $3, $4, $5, $6, 'indexing')
            RETURNING video_id;
        """

        async with self.pool.acquire() as conn:
            video_id = await conn.fetchval(sql, user_id, filename, video_path, duration, fps, resolution)

        return video_id

    async def _finalize_video_status(self, video_id: int, status: str) -> None:
        if not self.pool:
            return
        sql = "UPDATE videos SET processing_status = $1 WHERE video_id = $2"
        async with self.pool.acquire() as conn:
            await conn.execute(sql, status, video_id)

    async def _insert_events_batch(self, video_id: int, timestamps: Tuple[float, ...], captions: List[str], metas: List[Dict[str, int]], embeddings: np.ndarray) -> None:
        sql = """
            INSERT INTO video_events (video_id, timestamp, caption, yolo_metadata, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """

        data = []
        for i in range(len(timestamps)):
            # Преобразуем numpy array в список для безопасности, хотя register_vector позволяет использовать np.array
            vector_data = embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else embeddings[i]
            meta_json = json.dumps(metas[i])

            data.append((video_id, timestamps[i], captions[i], meta_json, vector_data))

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.executemany(sql, data)

    # --- ML PROCESSING (SYNCHRONOUS) ---

    def _check_motion_mog2(self, back_sub: cv2.BackgroundSubtractorMOG2, frame: np.ndarray, threshold: int) -> bool:
        fg = back_sub.apply(frame, learningRate=-1)
        fg = cv2.dilate(cv2.erode(fg, np.ones((3, 3), np.uint8)), np.ones((3, 3), np.uint8), iterations=2)
        return cv2.countNonZero(fg) > threshold

    def _process_yolo_batch(
        self, buffer: List[Tuple[np.ndarray, float]], output_queue: List[Tuple[Image.Image, float, Dict[str, int]]], prev_state: Optional[Dict[str, Any]], stats: Dict[str, int]
    ) -> Dict[str, Any]:
        frames = [x[0] for x in buffer]
        # YOLO inference
        results: List[Results] = self.yolo(frames, verbose=False, iou=0.7, conf=self.config.yolo_conf)

        current_prev = prev_state if prev_state else {"counts": {}, "centers": []}

        for i, res in enumerate(results):
            curr_state = self._extract_yolo_state(res)

            if self._has_state_changed(current_prev, curr_state):
                img_rgb = cv2.cvtColor(buffer[i][0], cv2.COLOR_BGR2RGB)
                output_queue.append((Image.fromarray(img_rgb), buffer[i][1], curr_state["counts"]))
                current_prev = curr_state
            else:
                stats["yolo_skip"] += 1

        return current_prev

    async def _process_florence_batch_and_save(self, queue: List[Tuple[Image.Image, float, Dict[str, int]]], video_id: int) -> int:
        if not queue:
            return 0

        images, timestamps, metas = zip(*queue)
        task = "<MORE_DETAILED_CAPTION>"

        try:
            # 1. Inference
            inputs = self.fl_processor(text=[task] * len(images), images=list(images), return_tensors="pt").to(self.device)
            if self.use_float16:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

            with torch.inference_mode():
                generated_ids = self.fl_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=256, num_beams=1, do_sample=False, use_cache=True)

            texts = self.fl_processor.batch_decode(generated_ids, skip_special_tokens=False)

            clean_texts = []
            for i, t in enumerate(texts):
                parsed = self.fl_processor.post_process_generation(t, task=task, image_size=images[i].size)
                clean_texts.append(parsed[task])

            embeddings = self.embedder.encode(clean_texts)

            # 2. Save
            await self._insert_events_batch(video_id, timestamps, clean_texts, list(metas), embeddings)

            return len(images)

        except Exception as e:
            print(f"Florence error: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def _extract_yolo_state(self, res: Results) -> Dict[str, Any]:
        """
        Извлекает состояние (кол-во объектов и их центры) из результата YOLO.
        """
        state: Dict[str, Any] = {"counts": {}, "centers": []}
        if res.boxes:
            for box in res.boxes:
                # Получаем имя класса
                lbl = res.names[int(box.cls[0])]
                state["counts"][lbl] = state["counts"].get(lbl, 0) + 1

                # Центры
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                state["centers"].append(((x1 + x2) / 2, (y1 + y2) / 2))

        # Сортируем центры для детерминированного сравнения
        state["centers"].sort()
        return state

    def _has_state_changed(self, prev: Dict[str, Any], curr: Dict[str, Any], threshold: int = 50) -> bool:
        """
        Проверяет, изменилось ли состояние объектов (кол-во или значительное смещение).
        """
        # Если количество или состав объектов изменился - считаем изменением
        if prev.get("counts") != curr.get("counts"):
            return True

        # Если объекты те же, проверяем, сдвинулись ли они
        p_centers = prev.get("centers", [])
        c_centers = curr.get("centers", [])

        if len(p_centers) == len(c_centers) and len(c_centers) > 0:
            # Вычисляем максимальное смещение среди всех пар объектов
            max_dist = max(((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** 0.5 for p, c in zip(p_centers, c_centers))
            if max_dist > threshold:
                return True

        return False

    def _setup_debug(self) -> None:
        if os.path.exists(self.config.debug_dir):
            shutil.rmtree(self.config.debug_dir)
        os.makedirs(self.config.debug_dir, exist_ok=True)


if __name__ == "__main__":

    async def main():
        # DSN для asyncpg
        dsn = "postgresql://postgres:пароль@localhost:5432/sceneseek_test"
        conf = IndexerConfig(db_dsn=dsn, frame_skip=15)

        engine = VideoSearchEngine(config=conf)
        await engine.initialize_db()

        try:
            # Убедитесь, что файл существует
            if os.path.exists("video.mp4"):
                print("\n Индексация...")
                await engine.run_indexing("video.mp4", user_id=1)

                print("\n Поиск...")
                results = await engine.search("a monkey")
                for res in results:
                    print(res)
            else:
                print("Файл video.mp4 не найден для теста.")
        finally:
            await engine.close()

    asyncio.run(main())
