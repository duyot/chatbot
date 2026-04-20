from pydantic import BaseModel
from uuid import UUID

class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    status: str

    model_config = {"from_attributes": True}
