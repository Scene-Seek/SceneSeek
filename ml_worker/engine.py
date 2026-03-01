import asyncio
import json
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
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
    """Configuration for video indexing pipeline"""

    frame_skip: int = 15
    yolo_batch_size: int = 16
    florence_batch_size: int = 4
    motion_threshold: int = 1000
    yolo_conf: float = 0.25
    db_dsn: str = "postgresql://postgres:password@localhost:5432/sceneseek_test"
    debug_mode: bool = False
    debug_dir: str = "debug_output"
    model_path: str = "yolov8n.pt"
    florence_model_id: str = "microsoft/Florence-2-base-ft"
    embedder_model: str = "all-MiniLM-L6-v2"

    @classmethod
    def from_settings(cls, settings):
        """Create IndexerConfig from Settings object"""
        return cls(
            frame_skip=settings.FRAME_SKIP,
            yolo_batch_size=settings.YOLO_BATCH_SIZE,
            florence_batch_size=settings.FLORENCE_BATCH_SIZE,
            motion_threshold=settings.MOTION_THRESHOLD,
            yolo_conf=settings.YOLO_CONF,
            db_dsn=settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql"),
            model_path=settings.MODEL_PATH,
            florence_model_id=settings.FLORENCE_MODEL_ID,
            embedder_model=settings.EMBEDDER_MODEL,
        )


class VideoSearchEngine:
    def __init__(self, config: Optional[IndexerConfig] = None, use_float16: bool = True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_float16 = use_float16 and self.device == "cuda"
        self.config = config if config else IndexerConfig()

        self.pool: Optional[asyncpg.Pool] = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # For async YOLO + other CPU-bound tasks
        print(f"Init] Starting engine on {self.device}...")
        self._load_models()

    async def _initialize_db(self) -> None:
        try:
            self.pool = await asyncpg.create_pool(dsn=self.config.db_dsn)

            if self.pool is None:
                raise ConnectionError("Failed to create database connection pool.")
        except Exception as e:
            print(f"[Database] Connection error: {e}")

    async def close(self) -> None:
        """Close database connection pool and executor."""
        if self.pool:
            await self.pool.close()
        if self.executor:
            self.executor.shutdown(wait=True)

    def _load_models(self) -> None:
        """Load YOLO, Florence, and embedding models."""
        print(" ├─ [1/3] YOLOv8...")
        self.yolo = YOLO(self.config.model_path)

        print(" ├─ [2/3] Florence-2...")
        dtype = torch.float16 if self.use_float16 else torch.float32
        self.fl_model = AutoModelForCausalLM.from_pretrained(self.config.florence_model_id, trust_remote_code=True, torch_dtype=dtype).to(self.device).eval()
        self.fl_processor = AutoProcessor.from_pretrained(self.config.florence_model_id, trust_remote_code=True)

        print(" ├─ [3/3] Embedder...")
        self.embedder = SentenceTransformer(self.config.embedder_model, device=self.device)
        print(" └─ Done.")

    # ====================== SEARCH OPERATIONS ======================

    async def search(self, query: str, *, query_id: Optional[int] = None, video_id: Optional[int] = None, top_k: int = 5, min_score: float = 0.25) -> List[Dict[str, Any]]:
        """Semantic search across indexed video events using query embedding.

        Args:
            query: Text query to search for
            query_id: Optional query ID for tracking results
            video_id: Optional video ID to restrict search to a specific video
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of matching events with metadata and scores
        """
        if not self.pool:
            raise RuntimeError("Database not initialized. Call _initialize_db() first.")

        # Encode query text to embedding
        query_vec: List[float] = self.embedder.encode(query).tolist()

        # Search database for similar events, optionally filtered by video_id
        if video_id is not None:
            sql = """
                SELECT
                    e.event_id,
                    v.video_id,
                    v.title,
                    e.timestamp,
                    e.caption,
                    1 - (e.embedding <=> $1) as score,
                    e.yolo_metadata
                FROM video_events e
                JOIN videos v ON e.video_id = v.video_id
                WHERE e.video_id = $3
                ORDER BY e.embedding <=> $1
                LIMIT $2;
            """
        else:
            sql = """
                SELECT
                    e.event_id,
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
        db_rows: List[Tuple[int, float, Optional[bool]]] = []
        try:
            async with self.pool.acquire() as conn:
                # Register vector type for current connection to pass query_vec correctly
                await register_vector(conn)
                if video_id is not None:
                    rows = await conn.fetch(sql, query_vec, top_k, video_id)
                else:
                    rows = await conn.fetch(sql, query_vec, top_k)

                for r in rows:
                    if r["score"] < min_score:
                        continue

                    # Parse YOLO metadata
                    meta = r["yolo_metadata"]
                    if isinstance(meta, str):
                        meta = json.loads(meta)

                    score = round(float(r["score"]), 4)
                    results.append({"video_id": r["video_id"], "video_title": r["title"], "timestamp": round(r["timestamp"], 2), "caption": r["caption"], "score": score, "metadata": meta})
                    db_rows.append((r["event_id"], score, None))
        except Exception as e:
            print(f"[Search] Error: {e}")

        if query_id is not None and db_rows:
            await self._insert_search_results(query_id=query_id, results=db_rows)

        return results

    # ====================== INDEXING OPERATIONS ======================

    async def run_indexing(self, video_path: str, user_id: int = 1, video_id: int | None = None) -> None:
        """Main video indexing pipeline: extract frames, run ML models, save to DB.

        Args:
            video_path: Path to video file
            user_id: User ID who uploaded the video
            video_id: Existing video ID from gateway (if provided, skip creating a new entry)
        """
        print("📹 [Indexing] Starting video processing...")
        if self.config.debug_mode:
            self._setup_debug()

        if not self.pool:
            await self._initialize_db()

        # Use existing video_id from gateway, or create a new entry as fallback
        if video_id is None:
            video_id = await self._create_video_entry(video_path, user_id)
        else:
            # Mark existing video as indexing
            await self._finalize_video_status(video_id, "indexing")

        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Motion detection via background subtraction
            back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

            yolo_buffer: List[Tuple[np.ndarray, float]] = []
            florence_queue: List[Tuple[Image.Image, float, Dict[str, int]]] = []

            # State: {'counts': dict, 'centers': list}
            prev_yolo_state: Optional[Dict[str, Any]] = None

            frame_idx = 0
            stats = {"motion_skip": 0, "yolo_skip": 0, "indexed": 0}

            while True:
                success, frame = cap.read()
                if not success:
                    break

                # Check for motion in frame
                is_motion = False
                if frame_idx % 5 == 0:
                    is_motion = self._check_motion_mog2(back_sub, frame, self.config.motion_threshold)

                # Process frames at configured interval
                if frame_idx % self.config.frame_skip == 0:
                    timestamp = frame_idx / fps

                    if not is_motion:
                        stats["motion_skip"] += 1
                        frame_idx += 1
                        continue

                    yolo_buffer.append((frame, timestamp))

                    # Process YOLO batch async (non-blocking)
                    if len(yolo_buffer) >= self.config.yolo_batch_size:
                        prev_yolo_state = await self._process_yolo_batch_async(yolo_buffer, florence_queue, prev_yolo_state, stats)
                        yolo_buffer = []
                        print(f" ⏳ {timestamp:.1f}s / {duration:.1f}s", end="\r")

                    # Process Florence batch and save to DB
                    if len(florence_queue) >= self.config.florence_batch_size:
                        count = await self._process_florence_batch_and_save(florence_queue, video_id)
                        stats["indexed"] += count
                        florence_queue = []

                frame_idx += 1

            # Process remaining buffered items
            if yolo_buffer:
                prev_yolo_state = await self._process_yolo_batch_async(yolo_buffer, florence_queue, prev_yolo_state, stats)
            if florence_queue:
                count = await self._process_florence_batch_and_save(florence_queue, video_id)
                stats["indexed"] += count

            await self._finalize_video_status(video_id, "ready")
            print(f"\n[Indexing] Completed. Saved {stats['indexed']} events.")

        except Exception as e:
            print(f"\n[Indexing] Error: {e}")
            await self._finalize_video_status(video_id, "failed")
            import traceback

            traceback.print_exc()
        finally:
            # Guaranteed resource cleanup
            if cap is not None:
                cap.release()
                cap = None
            yolo_buffer.clear()
            florence_queue.clear()

    # ====================== DATABASE OPERATIONS ======================

    async def _create_video_entry(self, video_path: str, user_id: int) -> int:
        """Create video entry in database and return video ID.

        Args:
            video_path: Path to video file
            user_id: User ID who uploaded

        Returns:
            video_id from database
        """
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            filename = os.path.basename(video_path)
            resolution = f"{width}x{height}"

            sql = """
                INSERT INTO videos
                  (uploaded_by_user_id, title, path, duration, fps, resolution, processing_status)
                VALUES ($1, $2, $3, $4, $5, $6, 'indexing')
                RETURNING video_id
            """

            async with self.pool.acquire() as conn:
                video_id = await conn.fetchval(sql, user_id, filename, video_path, duration, fps, resolution)
            return video_id
        except Exception as e:
            print(f"[Video Entry] Error: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()
                cap = None

    async def _finalize_video_status(self, video_id: int, status: str) -> None:
        """Update video processing status.

        Args:
            video_id: Video ID
            status: New status (ready, failed, indexing, etc.)
        """
        if not self.pool:
            return
        sql = "UPDATE videos SET processing_status = $1 WHERE video_id = $2"
        async with self.pool.acquire() as conn:
            await conn.execute(sql, status, video_id)

    async def update_search_status(self, query_id: int, query_text: str, status: str = "ready") -> None:
        """Update search_history with query embedding and processing status after search completes.

        Args:
            query_id: Search query ID
            query_text: Original query text
            status: Search status (ready, not_found, failed, etc.)
        """
        if not self.pool:
            return

        try:
            # Encode query to embedding
            query_embedding = self.embedder.encode(query_text).tolist()

            # Update search_history with embedding and status
            sql = """
                UPDATE search_history
                SET query_embedding = COALESCE(query_embedding, $1),
                    processing_status = $2
                WHERE query_id = $3
            """
            async with self.pool.acquire() as conn:
                await register_vector(conn)
                await conn.execute(sql, query_embedding, status, query_id)
        except Exception as e:
            print(f"[Search Status] Error: {e}")

    async def _insert_events_batch(self, video_id: int, timestamps: Tuple[float, ...], captions: List[str], metas: List[Dict[str, int]], embeddings: np.ndarray) -> None:
        """Insert batch of video events to database.

        Args:
            video_id: Video ID
            timestamps: Frame timestamps
            captions: Florence captions for frames
            metas: YOLO detection metadata
            embeddings: Sentence transformer embeddings
        """
        sql = """
            INSERT INTO video_events
              (video_id, timestamp, caption, yolo_metadata, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """

        data = []
        for i in range(len(timestamps)):
            # Convert numpy array to list for safety
            vector_data = embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else embeddings[i]
            meta_json = json.dumps(metas[i])
            data.append((video_id, timestamps[i], captions[i], meta_json, vector_data))

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.executemany(sql, data)

    async def _insert_search_results(self, *, query_id: int, results: List[Tuple[int, float, Optional[bool]]]) -> None:
        """Save search results to database.

        Args:
            query_id: Search query ID
            results: List of (event_id, similarity_score, is_relevant) tuples
        """
        if not self.pool or not results:
            return

        sql = """
            INSERT INTO search_results
              (query_id, found_event_id, similarity_score, is_relevant)
            VALUES ($1, $2, $3, $4)
        """

        data = [(query_id, event_id, score, is_relevant) for event_id, score, is_relevant in results]

        async with self.pool.acquire() as conn:
            await conn.executemany(sql, data)

    # ====================== ML PROCESSING OPERATIONS ======================

    def _check_motion_mog2(self, back_sub: cv2.BackgroundSubtractorMOG2, frame: np.ndarray, threshold: int) -> bool:
        """Detect motion in frame using background subtraction.

        Args:
            back_sub: OpenCV background subtractor
            frame: Video frame (BGR)
            threshold: Motion pixel count threshold

        Returns:
            True if motion detected, False otherwise
        """
        fg = back_sub.apply(frame, learningRate=-1)
        fg = cv2.dilate(cv2.erode(fg, np.ones((3, 3), np.uint8)), np.ones((3, 3), np.uint8), iterations=2)
        return cv2.countNonZero(fg) > threshold

    def _process_yolo_batch_sync(
        self, buffer: List[Tuple[np.ndarray, float]], output_queue: List[Tuple[Image.Image, float, Dict[str, int]]], prev_state: Optional[Dict[str, Any]], stats: Dict[str, int]
    ) -> Dict[str, Any]:
        """Synchronous YOLO object detection on frame batch.

        Args:
            buffer: List of (frame, timestamp) tuples
            output_queue: Output queue to append detected frames
            prev_state: Previous YOLO state for change detection
            stats: Statistics dict to update

        Returns:
            Current YOLO state after processing
        """
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

    async def _process_yolo_batch_async(
        self, buffer: List[Tuple[np.ndarray, float]], output_queue: List[Tuple[Image.Image, float, Dict[str, int]]], prev_state: Optional[Dict[str, Any]], stats: Dict[str, int]
    ) -> Dict[str, Any]:
        """Async wrapper for YOLO batch processing using ThreadPoolExecutor.

        Args:
            buffer: List of (frame, timestamp) tuples
            output_queue: Output queue to append detected frames
            prev_state: Previous YOLO state for change detection
            stats: Statistics dict to update

        Returns:
            Current YOLO state after processing
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_yolo_batch_sync, buffer, output_queue, prev_state, stats)

    async def _process_florence_batch_and_save(self, queue: List[Tuple[Image.Image, float, Dict[str, int]]], video_id: int) -> int:
        """Process images with Florence-2, generate embeddings, and save to DB.

        Args:
            queue: List of (image, timestamp, yolo_metadata) tuples
            video_id: Video ID for database insertion

        Returns:
            Number of frames processed successfully
        """
        if not queue:
            return 0

        images, timestamps, metas = zip(*queue)
        task = "<MORE_DETAILED_CAPTION>"

        try:
            # Prepare inputs for Florence-2 model
            inputs = self.fl_processor(text=[task] * len(images), images=list(images), return_tensors="pt").to(self.device)

            if self.use_float16:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

            # Generate captions
            with torch.inference_mode():
                generated_ids = self.fl_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=256, num_beams=1, do_sample=False, use_cache=True)

            # Decode and post-process captions
            texts = self.fl_processor.batch_decode(generated_ids, skip_special_tokens=False)

            clean_texts = []
            for i, t in enumerate(texts):
                parsed = self.fl_processor.post_process_generation(t, task=task, image_size=images[i].size)
                clean_texts.append(parsed[task])

            # Generate sentence embeddings
            embeddings = self.embedder.encode(clean_texts)

            # Save to database
            await self._insert_events_batch(video_id, timestamps, clean_texts, list(metas), embeddings)

            return len(images)

        except Exception as e:
            print(f"[Florence] Error: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def _extract_yolo_state(self, res: Results) -> Dict[str, Any]:
        """Extract object counts and centers from YOLO detection result.

        Args:
            res: YOLO detection result

        Returns:
            Dictionary with 'counts' (class counts) and 'centers' (sorted centers)
        """
        state: Dict[str, Any] = {"counts": {}, "centers": []}

        if res.boxes:
            for box in res.boxes:
                # Extract class name
                label = res.names[int(box.cls[0])]
                state["counts"][label] = state["counts"].get(label, 0) + 1

                # Extract box center
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                state["centers"].append((center_x, center_y))

        # Sort centers for deterministic comparison
        state["centers"].sort()
        return state

    def _has_state_changed(self, prev: Dict[str, Any], curr: Dict[str, Any], threshold: int = 50) -> bool:
        """Check if YOLO object state changed significantly.

        Args:
            prev: Previous state (counts, centers)
            curr: Current state (counts, centers)
            threshold: Distance threshold for considering movement

        Returns:
            True if object counts changed or objects moved significantly
        """
        # Check if object counts or composition changed
        if prev.get("counts") != curr.get("counts"):
            return True

        # Check if objects moved significantly
        prev_centers = prev.get("centers", [])
        curr_centers = curr.get("centers", [])

        if len(prev_centers) == len(curr_centers) and len(curr_centers) > 0:
            # Calculate maximum displacement among all object pairs
            max_distance = max(((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** 0.5 for p, c in zip(prev_centers, curr_centers))
            if max_distance > threshold:
                return True

        return False

    def _setup_debug(self) -> None:
        """Initialize debug output directory (removes existing if present)."""
        if os.path.exists(self.config.debug_dir):
            shutil.rmtree(self.config.debug_dir)
        os.makedirs(self.config.debug_dir, exist_ok=True)
