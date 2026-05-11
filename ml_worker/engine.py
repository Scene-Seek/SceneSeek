import asyncio
import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pgvector.asyncpg import register_vector
from PIL import Image
from stop_words import get_stop_words
from transformers import SiglipModel, SiglipProcessor
from ultralytics import YOLO

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class IndexerConfig:
    """Настройки пайплайна индексации и поиска по видео"""

    frame_skip: int = 15
    batch_size: int = 32
    motion_threshold: int = 100
    yolo_conf: float = 0.25
    model_path: str = "yolo26s.pt"
    siglip_model_id: str = "google/siglip2-base-patch16-224"
    db_dsn: str = "postgresql://postgres:password@localhost:5432/sceneseek_test"
    merge_threshold_sec: float = 2.0
    min_hits_in_segment: int = 4
    raw_search_limit: int = 5000
    sql_min_similarity: float = 0.05

    @classmethod
    def from_settings(cls, settings: Any) -> "IndexerConfig":
        """Строит конфигурацию из объекта настроек и подставляет значения по умолчанию"""
        return cls(
            frame_skip=getattr(settings, "FRAME_SKIP", cls.frame_skip),
            batch_size=getattr(settings, "BATCH_SIZE", cls.batch_size),
            motion_threshold=getattr(settings, "MOTION_THRESHOLD", cls.motion_threshold),
            yolo_conf=getattr(settings, "YOLO_CONF", cls.yolo_conf),
            model_path=getattr(settings, "MODEL_PATH", getattr(settings, "YOLO_MODEL", cls.model_path)),
            siglip_model_id=getattr(settings, "SIGLIP_MODEL_ID", getattr(settings, "CLIP_MODEL", cls.siglip_model_id)),
            raw_search_limit=getattr(settings, "RAW_SEARCH_LIMIT", cls.raw_search_limit),
            sql_min_similarity=getattr(settings, "SQL_MIN_SIMILARITY", cls.sql_min_similarity),
            db_dsn=settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql"),
        )


class ModelRuntime:
    """Жизненный цикл моделей, обработка кадров и векторизация"""

    def __init__(self, config: IndexerConfig, device: str) -> None:
        """Готовит окружение для инференса и базовые параметры рантайма"""
        self.config = config
        self.device = device
        self.detector: YOLO
        self.siglip_model: SiglipModel
        self.siglip_processor: SiglipProcessor
        self.stopwords = self._build_stopwords()

    def load_models(self) -> None:
        """Загружает модели детекции и текстово-визуальные энкодеры"""
        total_started = perf_counter()

        yolo_started = perf_counter()
        logger.info("Loading YOLO model: %s", self.config.model_path)
        self.detector = YOLO(self.config.model_path, verbose=False)
        logger.info("YOLO model loaded in %.2fs", perf_counter() - yolo_started)

        siglip_started = perf_counter()
        logger.info("Loading SigLIP model: %s", self.config.siglip_model_id)
        self.siglip_model = SiglipModel.from_pretrained(self.config.siglip_model_id).to(self.device).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained(self.config.siglip_model_id)
        logger.info("SigLIP model loaded in %.2fs", perf_counter() - siglip_started)
        logger.info("Engine initialization complete in %.2fs", perf_counter() - total_started)

    def append_frame_views(
        self,
        frame: np.ndarray,
        ts: float,
        img_buffer: List[Image.Image],
        meta_buffer: List[Dict[str, Any]],
    ) -> None:
        """Добавляет глобальный и локальные кропы кадра в буфер для векторизации"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        img_buffer.append(pil_img.resize((224, 224)))
        meta_buffer.append({
            "ts": ts,
            "bbox": [0, 0, frame.shape[1], frame.shape[0]],
            "type": "global",
        })

        results = self.detector(frame, verbose=False, conf=self.config.yolo_conf)[0]
        for box in results.boxes.xyxy:
            coords = box.tolist()
            crop = self.get_padding_crop(pil_img, coords)
            img_buffer.append(crop.resize((224, 224)))
            meta_buffer.append({"ts": ts, "bbox": coords, "type": "local"})

    def get_padding_crop(self, pil_img: Image.Image, box: List[float], padding: float = 0.25) -> Image.Image:
        """Вырезает кроп с паддингом вокруг бокса и ограничивает его границами изображения"""
        w, h = pil_img.size
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        return pil_img.crop((max(0, x1 - bw * padding), max(0, y1 - bh * padding), min(w, x2 + bw * padding), min(h, y2 + bh * padding)))

    def _extract_tensor(self, outputs: Any) -> torch.Tensor:
        """Достает тензор эмбеддингов из ответа модели в разных форматах"""
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def _build_stopwords(self) -> set[str]:
        """Собирает набор стоп-слов для русского и английского языков"""
        return set(get_stop_words("en")) | set(get_stop_words("ru"))

    def normalize_query(self, query: str) -> str:
        """Очищает запрос от пунктуации, приводит к нижнему регистру и убирает стоп-слова"""
        text = query.strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"[\s_]+", " ", text).strip()
        tokens = [token for token in text.split() if token not in self.stopwords]
        return " ".join(tokens) if tokens else text

    @torch.inference_mode()
    def vectorize_images_sync(self, images: List[Image.Image]) -> np.ndarray:
        """Синхронно строит нормализованные эмбеддинги для списка изображений"""
        if not images:
            return np.array([])
        inputs = self.siglip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.siglip_model.get_image_features(**inputs)
            features = self._extract_tensor(outputs)
            features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy()

    @torch.inference_mode()
    def vectorize_text_sync(self, query: str) -> np.ndarray:
        """Синхронно строит нормализованный эмбеддинг для текстового запроса"""
        normalized = self.normalize_query(query)
        prompt_query = normalized if normalized else query.strip().lower()
        full_query = f"a photo of {prompt_query}"
        inputs = self.siglip_processor(text=[full_query], return_tensors="pt", padding="max_length").to(self.device)
        outputs = self.siglip_model.get_text_features(**inputs)
        text_vec = self._extract_tensor(outputs)
        text_vec = F.normalize(text_vec, p=2, dim=-1)
        return text_vec.cpu().numpy()[0]


class SearchPostProcessor:
    """Постобработка результатов поиска: скоринг и сегментация"""

    def build_scored_hits(self, rows: List[asyncpg.Record]) -> List[Dict[str, Any]]:
        """Преобразует сырые строки БД в хиты, применяя динамический порог по скору"""
        parsed_rows: List[Dict[str, Any]] = []
        scores: List[float] = []
        for row in rows:
            meta = row["yolo_metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)

            score = float(row["score"])
            scores.append(score)

            parsed_rows.append({
                "event_id": int(row["event_id"]),
                "video_id": int(row["video_id"]),
                "video_title": row["title"],
                "ts": float(row["timestamp"]),
                "score": score,
                "bbox": meta.get("bbox", []),
                "type": meta.get("type", "unknown"),
            })

        if not parsed_rows:
            return []

        scores_np = np.asarray(scores, dtype=np.float32)
        mean_score = float(np.mean(scores_np))
        std_score = float(np.std(scores_np))

        base_threshold = mean_score + 0.5 * std_score
        keep_count = max(1, min(len(parsed_rows), max(8, int(np.ceil(len(parsed_rows) * 0.2)))))
        sorted_scores = np.sort(scores_np)
        keep_floor = float(sorted_scores[-keep_count])
        threshold = max(base_threshold, keep_floor * 0.8)

        logger.info(
            "Search score stats: candidates=%s mean=%.4f std=%.4f max=%.4f threshold=%.4f keep_floor=%.4f keep_count=%s",
            len(parsed_rows),
            mean_score,
            std_score,
            float(np.max(scores_np)),
            threshold,
            keep_floor,
            keep_count,
        )

        frame_best_hits: Dict[Tuple[int, float], Dict[str, Any]] = {}
        for hit in parsed_rows:
            if hit["score"] < threshold:
                continue
            key = (hit["video_id"], round(hit["ts"], 6))
            if key not in frame_best_hits or hit["score"] > frame_best_hits[key]["score"]:
                frame_best_hits[key] = hit.copy()

        hits = list(frame_best_hits.values())
        hits.sort(key=lambda item: (item["video_id"], item["ts"]))
        logger.info("Search hits retained after thresholding: kept=%s dropped=%s", len(hits), len(parsed_rows) - len(hits))
        return hits

    def cluster_hits(self, raw_hits: List[Dict[str, Any]], merge_threshold_sec: float) -> List[List[Dict[str, Any]]]:
        """Группирует хиты в сегменты по временному зазору внутри одного видео"""
        per_video_hits: Dict[int, List[Dict[str, Any]]] = {}
        for hit in raw_hits:
            per_video_hits.setdefault(hit["video_id"], []).append(hit)

        segments: List[List[Dict[str, Any]]] = []
        for hits in per_video_hits.values():
            current_segment = [hits[0]]
            for hit in hits[1:]:
                if hit["ts"] - current_segment[-1]["ts"] <= merge_threshold_sec:
                    current_segment.append(hit)
                else:
                    segments.append(current_segment)
                    current_segment = [hit]
            segments.append(current_segment)
        return segments

    def build_segment_results(self, segments: List[List[Dict[str, Any]]], min_segment_hits: int) -> List[Dict[str, Any]]:
        """Формирует итоговые сегменты, выбирая лучший хит и агрегируя время"""
        results: List[Dict[str, Any]] = []
        for segment in segments:
            if len(segment) < min_segment_hits:
                continue
            best_hit = max(segment, key=lambda item: item["score"])
            results.append({
                "event_id": best_hit["event_id"],
                "video_id": best_hit["video_id"],
                "video_title": best_hit["video_title"],
                "start": round(segment[0]["ts"], 2),
                "end": round(segment[-1]["ts"], 2),
                "best_ts": round(best_hit["ts"], 2),
                "score": round(best_hit["score"], 4),
                "bbox": best_hit["bbox"],
                "type": best_hit["type"],
            })
        return results


class VideoSearchEngine:
    def __init__(self, config: Optional[IndexerConfig] = None) -> None:
        """Инициализирует движок поиска, модели и пул подключений"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config if config else IndexerConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.runtime = ModelRuntime(config=self.config, device=self.device)
        self.search_post_processor = SearchPostProcessor()

        logger.info("Starting video search engine on device=%s", self.device)
        self._load_models()

    def _load_models(self) -> None:
        """Загружает модели в рантайм"""
        self.runtime.load_models()

    async def close(self) -> None:
        """Закрывает пул БД и останавливает потоковый пул"""
        if self.pool:
            logger.info("Closing asyncpg pool")
            await self.pool.close()
        self.executor.shutdown(wait=True)
        logger.info("Video search engine shutdown complete")

    async def _initialize_db(self) -> None:
        """Создает пул подключений к БД при первом использовании"""
        if not self.pool:
            logger.info("Initializing asyncpg pool inside engine")
            self.pool = await asyncpg.create_pool(dsn=self.config.db_dsn, init=register_vector)
            logger.info("Asyncpg pool initialized")

    # ====================== Индексирование ======================

    async def run_indexing(self, video_path: str, user_id: int = 1, video_id: int | None = None) -> None:
        """Индексирует видеофайл, создавая события и эмбеддинги в базе данных"""
        started = perf_counter()
        await self._initialize_db()

        if video_id is None:
            video_id = await self._create_video_entry(video_path, user_id)
        else:
            await self._finalize_video_status(video_id, "indexing")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)

        img_buffer: List[Image.Image] = []
        meta_buffer: List[Dict[str, Any]] = []
        frame_idx = 0
        indexed_count = 0

        logger.info(
            "Indexing started: video_id=%s user_id=%s file=%s fps=%.2f total_frames=%s frame_skip=%s batch_size=%s",
            video_id,
            user_id,
            os.path.basename(video_path),
            fps,
            total_frames,
            self.config.frame_skip,
            self.config.batch_size,
        )

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % self.config.frame_skip != 0:
                frame_idx += 1
                continue

            ts = frame_idx / fps
            indexed_count += await self._process_frame_for_indexing(video_id, frame, ts, back_sub, img_buffer, meta_buffer)

            frame_idx += 1
            if frame_idx % 500 == 0:
                logger.info(
                    "Indexing progress: video_id=%s processed_frames=%s total_frames=%s buffered_items=%s indexed_events=%s",
                    video_id,
                    frame_idx,
                    total_frames,
                    len(img_buffer),
                    indexed_count,
                )

        if img_buffer:
            await self._process_and_save_batch(video_id, img_buffer, meta_buffer)
            indexed_count += len(img_buffer)

        await self._finalize_video_status(video_id, "ready")
        cap.release()

        logger.info(
            "Indexing complete: video_id=%s saved_events=%s processed_frames=%s duration_sec=%.2f",
            video_id,
            indexed_count,
            frame_idx,
            perf_counter() - started,
        )

    async def _process_frame_for_indexing(
        self,
        video_id: int,
        frame: np.ndarray,
        ts: float,
        back_sub: cv2.BackgroundSubtractor,
        img_buffer: List[Image.Image],
        meta_buffer: List[Dict[str, Any]],
    ) -> int:
        """Обрабатывает один кадр и при необходимости сохраняет батч эмбеддингов"""
        if self._should_skip_by_motion(frame, back_sub):
            return 0

        self._append_frame_views(frame, ts, img_buffer, meta_buffer)

        if len(img_buffer) < self.config.batch_size:
            return 0

        saved_count = len(img_buffer)
        await self._process_and_save_batch(video_id, img_buffer, meta_buffer)
        img_buffer.clear()
        meta_buffer.clear()
        return saved_count

    def _should_skip_by_motion(self, frame: np.ndarray, back_sub: cv2.BackgroundSubtractor) -> bool:
        """Определяет, стоит ли пропустить кадр по низкой динамике"""
        small = cv2.resize(frame, (320, 240))
        return cv2.countNonZero(back_sub.apply(small)) < self.config.motion_threshold

    def _append_frame_views(
        self,
        frame: np.ndarray,
        ts: float,
        img_buffer: List[Image.Image],
        meta_buffer: List[Dict[str, Any]],
    ) -> None:
        """Добавляет представления кадра в общий буфер индексации"""
        self.runtime.append_frame_views(frame, ts, img_buffer, meta_buffer)

    def _get_padding_crop(self, pil_img: Image.Image, box: List[float], padding: float = 0.25) -> Image.Image:
        """Делегирует вырезание кропа с паддингом в рантайм моделей"""
        return self.runtime.get_padding_crop(pil_img, box, padding)

    async def _process_and_save_batch(self, video_id: int, images: List[Image.Image], metas: List[Dict[str, Any]]) -> None:
        """Векторизует изображения батчем и сохраняет их в базу"""
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(self.executor, self._vectorize_images_sync, images)

        sql = """
            INSERT INTO video_events (video_id, timestamp, caption, yolo_metadata, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """
        batch_data = []
        for vector, meta in zip(vectors, metas):
            yolo_metadata = {"bbox": meta["bbox"], "type": meta["type"]}
            batch_data.append((video_id, meta["ts"], None, json.dumps(yolo_metadata), vector.tolist()))

        async with self.pool.acquire() as conn:
            await conn.executemany(sql, batch_data)
        logger.info("Saved embedding batch: video_id=%s batch_size=%s", video_id, len(batch_data))

    def _extract_tensor(self, outputs: Any) -> torch.Tensor:
        """Проксирует извлечение эмбеддингов из выходов модели"""
        return self.runtime._extract_tensor(outputs)

    @torch.inference_mode()
    def _vectorize_images_sync(self, images: List[Image.Image]) -> np.ndarray:
        """Проксирует синхронную векторизацию изображений"""
        return self.runtime.vectorize_images_sync(images)

    @torch.inference_mode()
    def _vectorize_text_sync(self, query: str) -> np.ndarray:
        """Проксирует синхронную векторизацию текста"""
        return self.runtime.vectorize_text_sync(query)

    # ====================== Поиск ======================

    async def search(
        self,
        query: str,
        *,
        query_id: Optional[int] = None,
        video_id: Optional[int] = None,
        top_k: int = 5,
        merge_threshold: Optional[float] = None,
        min_hits_in_segment: Optional[int] = None,
        raw_limit: Optional[int] = None,
        sql_min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Выполняет поиск по эмбеддингам и возвращает топ результатов"""
        started = perf_counter()
        await self._initialize_db()
        logger.info(
            "Search started: query_id=%s video_id=%s query_len=%s top_k=%s raw_limit=%s",
            query_id,
            video_id,
            len(query),
            top_k,
            raw_limit if raw_limit is not None else self.config.raw_search_limit,
        )

        loop = asyncio.get_running_loop()
        text_vec = await loop.run_in_executor(self.executor, self._vectorize_text_sync, query)

        raw_limit_value = raw_limit if raw_limit is not None else self.config.raw_search_limit
        sql_min_similarity_value = sql_min_similarity if sql_min_similarity is not None else self.config.sql_min_similarity

        rows = await self._fetch_search_source_rows(
            video_id=video_id,
            query_vec=text_vec,
            min_similarity=sql_min_similarity_value,
            raw_limit=max(top_k, raw_limit_value),
        )
        logger.info("Search source rows fetched: query_id=%s row_count=%s", query_id, len(rows))
        if not rows:
            logger.info("Search finished with no source rows: query_id=%s duration_sec=%.2f", query_id, perf_counter() - started)
            return []

        raw_hits = self._build_scored_hits(rows)
        if not raw_hits:
            logger.info("Search finished with no hits above threshold: query_id=%s duration_sec=%.2f", query_id, perf_counter() - started)
            return []

        merge_threshold_sec = merge_threshold if merge_threshold is not None else self.config.merge_threshold_sec
        min_segment_hits = min_hits_in_segment if min_hits_in_segment is not None else self.config.min_hits_in_segment

        segments = self._cluster_hits(raw_hits, merge_threshold_sec)
        results = self._build_segment_results(segments, min_segment_hits)

        results.sort(key=lambda item: item["score"], reverse=True)
        final_results = results[:top_k]

        if query_id is not None:
            await self._insert_search_results(query_id=query_id, results=final_results)

        logger.info(
            "Search complete: query_id=%s video_id=%s result_count=%s duration_sec=%.2f",
            query_id,
            video_id,
            len(final_results),
            perf_counter() - started,
        )
        return final_results

    async def _fetch_search_source_rows(
        self,
        *,
        video_id: Optional[int],
        query_vec: np.ndarray,
        min_similarity: float,
        raw_limit: int,
    ) -> List[asyncpg.Record]:
        """Запрашивает из БД кандидатов по векторной близости"""
        sql = """
            SELECT
                e.event_id,
                e.video_id,
                v.title,
                e.timestamp,
                e.yolo_metadata,
                (1 - (e.embedding <=> $1::vector)) AS score
            FROM video_events e
            JOIN videos v ON e.video_id = v.video_id
            WHERE e.embedding IS NOT NULL
              AND ($2::int IS NULL OR e.video_id = $2)
              AND (1 - (e.embedding <=> $1::vector)) >= $3
            ORDER BY e.embedding <=> $1::vector
            LIMIT $4
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, query_vec.tolist(), video_id, float(min_similarity), int(raw_limit))

    def _build_scored_hits(self, rows: List[asyncpg.Record]) -> List[Dict[str, Any]]:
        """Строит список хитов с весами из строк поиска"""
        return self.search_post_processor.build_scored_hits(rows)

    def _cluster_hits(self, raw_hits: List[Dict[str, Any]], merge_threshold_sec: float) -> List[List[Dict[str, Any]]]:
        """Объединяет хиты в сегменты по временному порогу"""
        return self.search_post_processor.cluster_hits(raw_hits, merge_threshold_sec)

    def _build_segment_results(self, segments: List[List[Dict[str, Any]]], min_segment_hits: int) -> List[Dict[str, Any]]:
        """Формирует итоговые результаты по сегментам"""
        return self.search_post_processor.build_segment_results(segments, min_segment_hits)

    async def update_search_status(self, query_id: int, query_text: str, status: str = "ready") -> None:
        """Сохраняет эмбеддинг запроса и обновляет статус обработки"""
        await self._initialize_db()
        logger.info("Updating search status: query_id=%s status=%s", query_id, status)
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(self.executor, self._vectorize_text_sync, query_text)

        sql = """
            UPDATE search_history
            SET query_embedding = COALESCE(query_embedding, $1),
                processing_status = $2
            WHERE query_id = $3
        """
        async with self.pool.acquire() as conn:
            await conn.execute(sql, query_embedding.tolist(), status, query_id)

    async def _insert_search_results(self, *, query_id: int, results: List[Dict[str, Any]]) -> None:
        """Записывает результаты поиска в таблицу истории"""
        if not results:
            logger.info("Skipping search result insert because result set is empty: query_id=%s", query_id)
            return
        sql = """
            INSERT INTO search_results
              (query_id, found_event_id, similarity_score, segment_start, segment_end, best_ts, bbox, hit_type, is_relevant)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        data = [
            (
                query_id,
                int(item["event_id"]),
                float(item["score"]),
                float(item["start"]),
                float(item["end"]),
                float(item["best_ts"]),
                json.dumps(item.get("bbox", [])),
                item.get("type", "unknown"),
                None,
            )
            for item in results
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, data)
        logger.info("Inserted search results: query_id=%s row_count=%s", query_id, len(data))

    # ====================== БД Хелперы ======================

    async def _create_video_entry(self, video_path: str, user_id: int) -> int:
        """Создает запись о видео и фиксирует метаданные файла"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        async with self.pool.acquire() as conn:
            video_id = await conn.fetchval(
                """
                INSERT INTO videos
                  (uploaded_by_user_id, title, path, duration, fps, resolution, processing_status)
                VALUES ($1, $2, $3, $4, $5, $6, 'indexing')
                RETURNING video_id
                """,
                user_id,
                os.path.basename(video_path),
                video_path,
                duration,
                fps,
                f"{width}x{height}",
            )
        logger.info(
            "Created video entry inside worker: video_id=%s user_id=%s title=%s resolution=%sx%s duration_sec=%.2f fps=%.2f",
            video_id,
            user_id,
            os.path.basename(video_path),
            width,
            height,
            duration,
            fps or 0.0,
        )
        return video_id

    async def _finalize_video_status(self, video_id: int, status: str) -> None:
        """Обновляет статус видео в базе после обработки"""
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE videos SET processing_status = $1 WHERE video_id = $2", status, video_id)
        logger.info("Updated video status: video_id=%s status=%s", video_id, status)
