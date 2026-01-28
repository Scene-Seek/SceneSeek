from faststream.rabbit import RabbitBroker

from src.core.broker import broker

class BrokerService():
    def __init__(self, broker: RabbitBroker):
        self.broker = broker
        self.queue = "videos"

    async def pub(self, *, message: str, queue: str):
        async with self.broker as br:
            await br.publish(
                message=message,
                queue=queue
            )

broker_service = BrokerService(broker=broker)