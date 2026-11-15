from contextlib import asynccontextmanager

from fastapi import FastAPI

from .auth import router as auth_router
from .db import db, init_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_indexes()
    yield


app = FastAPI(title="DiploMind API", lifespan=lifespan)
app.include_router(auth_router)


@app.get("/api/health")
def health():
    db.command("ping")  # fails if MongoDB is unreachable
    return {"status": "ok"}