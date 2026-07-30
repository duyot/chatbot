import logging
import os
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .observability import configure_ai_trace, install_record_factory
from .routers import documents
from .routers import chat
from .routers import auth
from .routers import conversations

# Must precede basicConfig: the format below reads %(trace_id)s off every
# record, and the factory is what guarantees the attribute exists.
install_record_factory()

_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(trace_id)s] %(name)s %(message)s"

logging.basicConfig(
    level=settings.log_level.upper(),
    format=_LOG_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
)
os.makedirs("/app/logs", exist_ok=True)
_fh = RotatingFileHandler("/app/logs/backend.log", maxBytes=10_485_760, backupCount=5)
_fh.setFormatter(logging.Formatter(_LOG_FORMAT))
logging.getLogger().addHandler(_fh)
# Separate JSONL sink for AI payloads; see app/observability.py.
configure_ai_trace()
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logger.info("%s %s %s", request.method, request.url.path, response.status_code)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(conversations.router)

@app.get("/health")
def health():
    return {"status": "ok"}
