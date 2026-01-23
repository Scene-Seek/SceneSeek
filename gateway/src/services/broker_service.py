from faststream.rabbit import RabbitBroker

from src.core.broker import broker

class BrokerService():
    def __init__(self, broker: RabbitBroker):
        self.broker = broker

    async def pub(self, *, message: str, queue: str):
        async with self.broker as br:
            br.publish(
                message=message,
                queue=queue
            )

broker_service = BrokerService(broker=broker)