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
    mime_type: str | None = None

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    document_id: str
    message: str
    conversation_id: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    username: str

    model_config = {"from_attributes": True}


class ConversationListItem(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    citations: list | None = None
    document_id: UUID | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    messages: list[MessageResponse]

    model_config = {"from_attributes": True}
