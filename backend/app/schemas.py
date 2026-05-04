from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    status: str

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    id: UUID
    file_name: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    document_id: str
    message: str
