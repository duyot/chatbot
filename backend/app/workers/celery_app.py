from celery import Celery
from ..config import settings

celery_app = Celery(
    "chatbot",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
