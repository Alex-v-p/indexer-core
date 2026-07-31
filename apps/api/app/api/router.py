from fastapi import APIRouter

from app.api.routes import documents, health, jobs, pipelines, queries

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(jobs.router)
api_router.include_router(pipelines.router)
api_router.include_router(queries.router)
