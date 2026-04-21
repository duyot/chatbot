Help me to planout the feature below:
1. from the chat ui, add the upload button in the chat box, to allow user to upload a file (docx, pdf, image)
2. Upon user uploaded, store document metadata (id, file name, path, uploaded date) to the postgresql db (make connection string configurable)
3. After store to database, run a background job, that ingest the document. this is to prepare for the RAG later. the ingest document should include:
chunking, embeddeding, store to pgvector.
4. after finish the ingestion, should prompt to the user that upload and ingested has been completed.