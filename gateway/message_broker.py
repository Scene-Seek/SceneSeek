from faststream.rabbit import RabbitBroker

broker = RabbitBroker()

async def send_message(message: str):
    """
    Публикует сообщение в RabbitMQ
    """
    await broker.publish(
        f"{message=}",
        queue=""
    )
    return {
        "data": "Success"
    }