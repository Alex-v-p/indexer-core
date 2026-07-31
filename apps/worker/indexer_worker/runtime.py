from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from packages.indexer_bootstrap.config import Settings
from indexer_worker.dispatcher import BackgroundJobDispatcher
from packages.indexer_application.dto import BackgroundJobRecord, BackgroundJobStatus, BackgroundJobType
from packages.indexer_application.services.background_jobs import prepared_document_from_payload
from packages.indexer_infrastructure.postgres.session import PostgresSessionManager
from packages.indexer_infrastructure.postgres.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)


class BackgroundWorkerRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._manager = PostgresSessionManager(
            database_url=settings.database_url,
            pool_size=max(
                settings.db_pool_size,
                settings.background_worker_concurrency * 2 + 1,
            ),
            max_overflow=settings.db_max_overflow,
            echo=settings.db_echo,
        )
        self._dispatcher = BackgroundJobDispatcher(settings)
        self._stop = asyncio.Event()
        hostname = socket.gethostname()
        self._worker_id = f"{hostname}:{uuid.uuid4().hex[:12]}"

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        logger.info(
            "Background worker started.",
            extra={
                "worker_id": self._worker_id,
                "concurrency": self._settings.background_worker_concurrency,
            },
        )
        loops = [
            asyncio.create_task(self._worker_loop(slot), name=f"worker-slot-{slot}")
            for slot in range(self._settings.background_worker_concurrency)
        ]
        await self._stop.wait()
        for task in loops:
            task.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        await self._manager.close()
        logger.info("Background worker stopped.", extra={"worker_id": self._worker_id})

    async def _worker_loop(self, slot: int) -> None:
        while not self._stop.is_set():
            try:
                job = await self._claim_next()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Background worker failed to claim a job.",
                    extra={"worker_id": self._worker_id, "worker_slot": slot},
                )
                await self._wait_for_poll_interval()
                continue

            if job is None:
                await self._wait_for_poll_interval()
                continue
            logger.info(
                "Background job claimed.",
                extra={
                    "worker_id": self._worker_id,
                    "worker_slot": slot,
                    "job_id": str(job.id),
                    "job_type": job.job_type.value,
                    "attempt": job.attempts,
                },
            )
            try:
                if job.current_stage == "lease_expired":
                    await self._record_abandoned(job)
                else:
                    await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Background worker could not persist the final job state.",
                    extra={
                        "worker_id": self._worker_id,
                        "worker_slot": slot,
                        "job_id": str(job.id),
                        "job_type": job.job_type.value,
                    },
                )
                await self._wait_for_poll_interval()

    async def _wait_for_poll_interval(self) -> None:
        try:
            await asyncio.wait_for(
                self._stop.wait(),
                timeout=self._settings.background_worker_poll_interval_seconds,
            )
        except TimeoutError:
            pass

    async def _claim_next(self) -> BackgroundJobRecord | None:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=self._settings.background_worker_lock_timeout_seconds)
        session_factory = self._manager.session_factory()
        async with session_factory() as session:
            uow = SqlAlchemyUnitOfWork(session)
            job = await uow.background_jobs.claim_next(
                worker_id=self._worker_id,
                now=now,
                stale_before=stale_before,
            )
            await uow.commit()
            return job

    async def _process(self, job: BackgroundJobRecord) -> None:
        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(job.id, heartbeat_stop),
            name=f"heartbeat-{job.id}",
        )
        try:
            session_factory = self._manager.session_factory()
            async with session_factory() as session:
                uow = SqlAlchemyUnitOfWork(session)

                async def report(progress: float, stage: str) -> None:
                    await self._update_progress(job.id, progress=progress, stage=stage)

                result = await self._dispatcher.dispatch(job=job, uow=uow, report=report)
                await uow.background_jobs.mark_succeeded(
                    job_id=job.id,
                    worker_id=self._worker_id,
                    result=result,
                    completed_at=datetime.now(UTC),
                )
                await uow.commit()
            logger.info(
                "Background job succeeded.",
                extra={"job_id": str(job.id), "job_type": job.job_type.value},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Background job failed.",
                extra={"job_id": str(job.id), "job_type": job.job_type.value},
            )
            await self._record_failure(job, exc)
        finally:
            heartbeat_stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _record_failure(self, job: BackgroundJobRecord, exc: Exception) -> None:
        failed_at = datetime.now(UTC)
        retry_at = None
        if job.attempts < job.max_attempts:
            delay = self._settings.background_worker_retry_base_seconds * (2 ** max(job.attempts - 1, 0))
            retry_at = failed_at + timedelta(seconds=delay)
        await self._persist_failure(
            job,
            error_message=str(exc),
            failed_at=failed_at,
            retry_at=retry_at,
        )

    async def _record_abandoned(self, job: BackgroundJobRecord) -> None:
        message = "Worker lease expired after the final permitted attempt."
        logger.error(
            message,
            extra={"job_id": str(job.id), "job_type": job.job_type.value},
        )
        await self._persist_failure(
            job,
            error_message=message,
            failed_at=datetime.now(UTC),
            retry_at=None,
        )

    async def _persist_failure(
        self,
        job: BackgroundJobRecord,
        *,
        error_message: str,
        failed_at: datetime,
        retry_at: datetime | None,
    ) -> None:
        session_factory = self._manager.session_factory()
        async with session_factory() as session:
            uow = SqlAlchemyUnitOfWork(session)
            persisted = await uow.background_jobs.mark_failed(
                job_id=job.id,
                worker_id=self._worker_id,
                error_message=error_message,
                failed_at=failed_at,
                retry_at=retry_at,
            )
            if (
                persisted.status is BackgroundJobStatus.FAILED
                and job.job_type is BackgroundJobType.INGEST_DOCUMENT
            ):
                try:
                    prepared = prepared_document_from_payload(job.payload)
                except Exception:
                    logger.exception(
                        "Terminal ingestion job payload could not be mapped to its document.",
                        extra={"job_id": str(job.id)},
                    )
                else:
                    await uow.documents.mark_failed(
                        document_id=prepared.document_id,
                        version_id=prepared.version.id,
                        error_message=error_message,
                    )
            await uow.commit()

    async def _heartbeat_loop(self, job_id: uuid.UUID, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self._settings.background_worker_heartbeat_seconds,
                )
                return
            except TimeoutError:
                pass
            try:
                session_factory = self._manager.session_factory()
                async with session_factory() as session:
                    uow = SqlAlchemyUnitOfWork(session)
                    await uow.background_jobs.heartbeat(
                        job_id=job_id,
                        worker_id=self._worker_id,
                        now=datetime.now(UTC),
                    )
                    await uow.commit()
            except Exception:
                logger.warning(
                    "Background job heartbeat failed.",
                    exc_info=True,
                    extra={"job_id": str(job_id)},
                )

    async def _update_progress(self, job_id: uuid.UUID, *, progress: float, stage: str) -> None:
        session_factory = self._manager.session_factory()
        async with session_factory() as session:
            uow = SqlAlchemyUnitOfWork(session)
            await uow.background_jobs.update_progress(
                job_id=job_id,
                worker_id=self._worker_id,
                progress=progress,
                current_stage=stage,
            )
            await uow.commit()
