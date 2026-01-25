Вот подробный `README.md`, составленный специально для твоего Олежки-бэкендера.

---

# 🧠 ML Video Search Engine (Indexer)

Этот модуль реализует логику "умного" индексирования видео. Он превращает видеофайл в набор семантических событий (Time + Description + Vector), по которым можно осуществлять текстовый поиск.

## 🚀 Описание архитектуры

Модуль работает как **черный ящик**, который подключается к PostgreSQL.

*   **Вход:** Путь к видеофайлу на диске (или S3 маунте).
*   **Процесс:**
    1.  **Motion Filter (MOG2):** Отсекает статические кадры.
    2.  **Object Filter (YOLOv8):** Отсекает движение "теней/света", оставляя только объекты (люди, машины, товары).
    3.  **Captioning (Florence-2):** Генерирует текстовое описание для отобранных кадров.
    4.  **Embedding (MiniLM):** Превращает текст в вектор (384 dim).
*   **Выход:** Записи в таблице `video_events` в PostgreSQL.

---

## 🔌 Интеграция с Бэкендом

### 1. Основной класс `VideoSearchEngine`

Инициализация движка происходит один раз при старте воркера (загрузка моделей в VRAM занимает время).

```python
from engine import VideoSearchEngine, IndexerConfig

# Конфигурация подключения
db_dsn = "postgresql://user:pass@host:5432/db_name"
config = IndexerConfig(db_dsn=db_dsn, frame_skip=15)

# Инициализация (Singleton на процесс)
engine = VideoSearchEngine(config=config)

# Обязательно инициализировать пул коннектов перед работой
await engine.initialize_db()
```

### 2. Ключевые методы

#### `run_indexing(video_path: str, user_id: int)`

Основной метод для **RabbitMQ воркера**.

*   **Что делает:** Прогоняет видео через ML-пайплайн и пишет результаты в БД.
*   **Блокирующий?** ML-часть блокирует Event Loop (CPU bound), запись в БД асинхронна.
*   **Результат:** Заполняет таблицы `videos` и `video_events`.

#### `search(query: str, top_k: int)`

Метод для **API поиска**.

*   **Что делает:** Превращает текст запроса в вектор и делает `ORDER BY cosine_distance` в базе.
*   **Скорость:** Очень быстро (ms).
*   **Результат:** Список словарей с таймкодами и скорами.

---

## 🐰 Интеграция с RabbitMQ (Worker)

Поскольку обработка видео длительная, ее нужно выносить в фоновый процесс.

**Пример логики воркера (consumer):**

```python
import asyncio
import aio_pika
from engine import VideoSearchEngine, IndexerConfig

async def main():
    # 1. Загружаем ML движок
    engine = VideoSearchEngine(config=IndexerConfig(db_dsn="..."))
    await engine.initialize_db()

    # 2. Подключаемся к RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
    queue_name = "video_indexing_tasks"

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(queue_name, durable=True)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    # Парсим задачу
                    data = json.loads(message.body)
                    video_path = data['path']
                    user_id = data['user_id']

                    print(f"Processing {video_path}...")

                    # 3. ЗАПУСКАЕМ ИНДЕКСАЦИЮ
                    # Важно: Воркер должен иметь доступ к файлу по этому пути
                    await engine.run_indexing(video_path, user_id=user_id)

                    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏛 База данных и SQLAlchemy

Мой движок использует **`asyncpg`** для максимальной скорости вставки векторов (bulk insert). Однако, твой бэкенд (API, админка) скорее всего использует **SQLAlchemy**.

**Важно:**

1.  Движок сам создает таблицы при первом запуске (`CREATE TABLE IF NOT EXISTS`).
2.  Ты можешь описать эти же таблицы в SQLAlchemy моделях (`models.py`), чтобы читать из них данные.

### Схема таблиц (для маппинга в SQLAlchemy)

#### Таблица `videos`

Хранит метаданные файла и статус обработки.

*   `video_id` (PK, Serial)
*   `path` (String)
*   `processing_status` (String): `'indexing' | 'ready' | 'failed'`

#### Таблица `video_events`

Хранит результаты ML. Это главная таблица для поиска.
*   `event_id` (PK, BigSerial)
*   `video_id` (FK -> videos)
*   `timestamp` (Float): секунда видео.
*   `caption` (Text): описание ("person holding a cup").
*   `yolo_metadata` (JSONB): счетчики объектов `{"person": 1, "car": 2}`.
*   `embedding` (Vector, 384 dim): Вектор описания.

### Пример SQLAlchemy модели (для чтения)

Тебе понадобится пакет `pgvector`.

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class VideoEvent(Base):
    __tablename__ = 'video_events'

    event_id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.video_id'))
    timestamp = Column(Float)
    caption = Column(Text)
    yolo_metadata = Column(JSONB)
    embedding = Column(Vector(384)) # Тип из pgvector

    # Пример гибридного поиска (SQLAlchemy):
    # session.query(VideoEvent).filter(VideoEvent.embedding.cosine_distance(query_vec) < 0.5).all()
```

---

## 💡 Рекомендации

1.  **Shared Storage:** Воркер (ML) и Бэкенд (API) должны иметь доступ к одной файловой системе. Если они в разных Docker-контейнерах, используйте общий Volume. В RabbitMQ передавайте путь к файлу внутри этого Volume.
2.  **Concurrency:** ML-модели (особенно Florence-2) занимают много VRAM. Не запускай много параллельных задач индексации на одной GPU. Лучше `prefetch_count=1` в RabbitMQ.
3.  **Migration:** Если нужно изменить схему БД, лучше делать это через Alembic на стороне бэкенда, а ML-коде просто поправить DDL. Мой скрипт использует `IF NOT EXISTS`, так что он не сломает твои таблицы.