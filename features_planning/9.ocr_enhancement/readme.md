# Task: Enhance the OCR process

## Context
1. OCR is process by calling to D:\development\chatbot\ocr-service, which is using rapidocr
2. the result is pure text but missing the understanding of table, header..., so the chunking is not so efficient
3. the application is running on local machine without GPU support.
## Requirements
### 1. check in huggingface or internet if any opensource provide better OCR
### 2. Integration into D:\development\chatbot\ocr-service, so the api can return more OCR data
### 2. Enhance the chunking process by Semantic Chunking (D:\development\chatbot\features_planning\9.ocr_enhancement\Contextual_Chunk_Headers.md)