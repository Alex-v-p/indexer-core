from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.dependencies.database import check_database_connection
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness probe that does not depend on external services."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(response: Response, settings: Settings = Depends(get_settings)) -> ReadinessResponse:
    """Readiness probe that verifies the database is reachable."""

    try:
        await check_database_connection(settings)
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", database="unavailable")

    return ReadinessResponse(status="ready", database="ok")
