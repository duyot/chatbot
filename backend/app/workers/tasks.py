from .celery_app import celery_app

@celery_app.task(bind=True, max_retries=1, default_retry_delay=10)
def ingest_document(self, document_id: str):
    pass  # implemented in Task 7
