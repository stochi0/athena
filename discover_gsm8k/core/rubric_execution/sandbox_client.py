"""Sandbox client: get/create sandbox, run commands."""

from __future__ import annotations

import asyncio
from typing import Any

from verifiers.envs.experimental.sandbox_mixin import SandboxMixin
from verifiers.envs.sandbox_env import CreateSandboxRequest


def _default_request() -> CreateSandboxRequest:
    return CreateSandboxRequest(
        name="discover-gsm8k-eval",
        docker_image="python:3.11-slim",
        start_command="tail -f /dev/null",
        cpu_cores=2,
        memory_gb=4,
        disk_size_gb=5,
        gpu_count=0,
        timeout_minutes=30,
        environment_vars={},
    )


class RubricSandboxClient(SandboxMixin):
    def __init__(self) -> None:
        self.init_sandbox_client(
            sandbox_client_max_workers=10,
            sandbox_client_max_connections=100,
            sandbox_client_max_keepalive_connections=50,
        )
        self._sandbox_id: str | None = None
        self._sandbox_lock = asyncio.Lock()

    async def get_or_create_sandbox(self) -> str:
        async with self._sandbox_lock:
            if self._sandbox_id is not None:
                return self._sandbox_id
            self._sandbox_id = await self.create_sandbox({}, _default_request())
            return self._sandbox_id

    async def execute_command(
        self, sandbox_id: str, command: str, *, timeout: int
    ) -> Any:
        return await self.with_retry(self.sandbox_client.execute_command)(
            sandbox_id, command, timeout=timeout
        )


_client: RubricSandboxClient | None = None


def get_sandbox_client() -> RubricSandboxClient:
    global _client
    if _client is None:
        _client = RubricSandboxClient()
    return _client
