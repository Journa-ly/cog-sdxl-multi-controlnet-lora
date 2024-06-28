import os
from celery import Celery, shared_task
from predict import Predictor

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "5672")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

app = Celery(
    'tasks',
    broker=f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}"
)

@shared_task(bind=True, queue="gqu_enabled_queue")
def generate_image(self, payload, **kwargs):
    predictor = Predictor()
    output = predictor.predict(**payload)
    return output