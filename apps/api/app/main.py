from fastapi import FastAPI

from app.api.router import api_router
from app.composition import build_query_pipeline_registry, build_query_tool_registry
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings: Settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.query_tool_registry = build_query_tool_registry(settings)
    app.state.query_pipeline_registry = build_query_pipeline_registry(
        settings,
        tool_registry=app.state.query_tool_registry,
    )

    register_exception_handlers(app, settings)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "status": "ok",
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
