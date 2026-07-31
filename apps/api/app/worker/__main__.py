from __future__ import annotations

import asyncio
import signal

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.worker.runtime import BackgroundWorkerRuntime


async def _run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    runtime = BackgroundWorkerRuntime(settings)
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, runtime.stop)
        except (NotImplementedError, RuntimeError):
            pass
    await runtime.run()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
