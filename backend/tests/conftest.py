import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("UPLOAD_DIR", "/tmp/test-uploads")

from app.main import app
from app.database import Base, get_db

TEST_DB_URL = os.environ["DATABASE_URL"]
engine = create_engine(TEST_DB_URL)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db():
    conn = engine.connect()
    trans = conn.begin()
    session = TestingSession(bind=conn)
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
