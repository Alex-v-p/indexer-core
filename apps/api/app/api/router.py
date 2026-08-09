from fastapi import APIRouter

from app.api.routes import document_organization, documents, health, jobs, pipelines, queries, subjects

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(jobs.router)
api_router.include_router(pipelines.router)
api_router.include_router(queries.router)
api_router.include_router(subjects.router)
api_router.include_router(document_organization.router)
