from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str = Field(examples=["ready"])
    database: str = Field(examples=["ok"])
