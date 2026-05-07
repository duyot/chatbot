import logging

from celery import Celery
from celery.signals import worker_ready
from ..config import settings

celery_app = Celery(
    "chatbot",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.worker_log_format = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
celery_app.conf.worker_task_log_format = "%(asctime)s %(levelname)-8s %(task_name)s[%(task_id)s]: %(message)s"


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logging.getLogger(__name__).info(
        "Celery worker ready — registered tasks: %s",
        list(sender.app.tasks.keys()),
    )
