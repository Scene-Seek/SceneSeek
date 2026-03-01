from typing import Any

from faststream.rabbit import RabbitBroker
from src.core.broker import broker


class BrokerService:
    def __init__(self, broker: RabbitBroker):
        self.broker = broker
        self.QUEUE_VIDEOS = "videos"
        self.QUEUE_SEARCHES = "searches"

    async def pub(self, *, message: Any, queue: str):
        """
        Publish message using the already-connected broker.
        """
        await self.broker.publish(message=message, queue=queue)


broker_service = BrokerService(broker=broker)
