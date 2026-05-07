import logging
import os
from logging.handlers import RotatingFileHandler

from celery import Celery
from celery.signals import after_setup_logger, worker_ready
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


@after_setup_logger.connect
def add_file_handler(logger, **kwargs):
    os.makedirs("/app/logs", exist_ok=True)
    fh = RotatingFileHandler("/app/logs/worker.log", maxBytes=10_485_760, backupCount=5)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))
    logger.addHandler(fh)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logging.getLogger(__name__).info(
        "Celery worker ready — registered tasks: %s",
        list(sender.app.tasks.keys()),
    )
