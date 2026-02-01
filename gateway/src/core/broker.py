"""broker"""

from faststream.rabbit import RabbitBroker

from src.core.config import settings

broker = RabbitBroker(url=settings.RABBITMQ_URL)