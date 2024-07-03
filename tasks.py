import os
import requests
from celery import Celery, shared_task

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
    print("Payload:", payload)

    response = requests.post("http://127.0.0.1:5000/predictions", json=payload)
    return response.json()