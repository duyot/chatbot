import logging
import os
from logging.handlers import RotatingFileHandler

from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger, worker_ready
from ..config import settings
from ..observability import configure_ai_trace, install_record_factory

# Both log formats below read %(trace_id)s; the factory guarantees the attribute
# exists on records from Celery internals and third-party libraries too.
install_record_factory()

_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(trace_id)s] %(name)s %(message)s"

celery_app = Celery(
    "chatbot",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.worker_log_format = _LOG_FORMAT
celery_app.conf.worker_task_log_format = (
    "%(asctime)s %(levelname)-8s [%(trace_id)s] %(task_name)s[%(task_id)s]: %(message)s"
)


@after_setup_logger.connect
def add_file_handler(logger, **kwargs):
    os.makedirs("/app/logs", exist_ok=True)
    fh = RotatingFileHandler("/app/logs/worker.log", maxBytes=10_485_760, backupCount=5)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(fh)
    # Separate JSONL sink for AI payloads; see app/observability.py.
    configure_ai_trace()


@after_setup_task_logger.connect
def add_task_file_handler(logger, **kwargs):
    """The task logger is configured separately from the worker logger and does
    not inherit its handlers, so per-task records would otherwise never reach
    worker.log."""
    os.makedirs("/app/logs", exist_ok=True)
    fh = RotatingFileHandler("/app/logs/worker.log", maxBytes=10_485_760, backupCount=5)
    fh.setFormatter(logging.Formatter(celery_app.conf.worker_task_log_format))
    logger.addHandler(fh)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logging.getLogger(__name__).info(
        "Celery worker ready — registered tasks: %s",
        list(sender.app.tasks.keys()),
    )
