import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("UPLOAD_DIR", "/tmp/test-uploads")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from app.main import app
from app.database import Base, get_db

TEST_DB_URL = os.environ["DATABASE_URL"]

# Guardrail: the session-scoped fixture below runs Base.metadata.drop_all at
# teardown, which will wipe whatever DB this URL points at. If you run pytest
# inside a container whose .env points at the production DB, that's a
# disaster — refuse to start instead. The DB name must contain "test".
_db_name = TEST_DB_URL.rsplit("/", 1)[-1].split("?", 1)[0].lower()
if "test" not in _db_name:
    raise RuntimeError(
        f"Refusing to run tests: DATABASE_URL points at {_db_name!r}, which "
        "does not contain 'test'. Set DATABASE_URL to a dedicated test "
        "database before running pytest, or your data will be dropped."
    )

engine = create_engine(TEST_DB_URL)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_tables():
    Base.metadata.create_all(bind=engine)
    # chunks_bm25 is migration-only: create_all() cannot build a pg_search
    # bm25 index, and declaring it on the model would make create_all() fail
    # outright on a plain Postgres without the extension. Create it here when
    # pg_search is available so the BM25 code path is exercisable in tests.
    with engine.begin() as conn:
        if conn.execute(
            sa_text("SELECT 1 FROM pg_extension WHERE extname = 'pg_search'")
        ).first():
            conn.execute(sa_text(
                "CREATE INDEX IF NOT EXISTS chunks_bm25 ON document_chunks "
                "USING bm25 (id, search_text) WITH (key_field = 'id')"
            ))
        # Plain GIN index backing the ts_rank fallback (migration 0011) — no
        # extension guard needed, this is stock Postgres full-text search.
        conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_search_text_fts "
            "ON document_chunks USING GIN (to_tsvector('english', search_text))"
        ))
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db():
    conn = engine.connect()
    trans = conn.begin()
    session = TestingSession(bind=conn, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    trans.rollback()
    conn.close()

@pytest.fixture
def client(db):
    def override_get_db():
        yield db
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
