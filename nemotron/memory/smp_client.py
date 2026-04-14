"""Async SMP client for the Nemotron agent.

Wraps the Structural Memory Protocol JSON-RPC API so the agent can query
and update the codebase graph without knowing transport details.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class SMPError(Exception):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(f"SMP error {code}: {message}")


class SMPClient:
    """Thin async client for the SMP JSON-RPC 2.0 server."""

    def __init__(self, base_url: str = "http://localhost:8420", timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None
        self._req_id = 0
        self._connected = False

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the SMP server, return True if healthy."""
        self._http = httpx.AsyncClient(base_url=self._base, timeout=self._timeout)
        try:
            r = await self._http.get("/health")
            self._connected = r.status_code == 200 and r.json().get("status") == "ok"
        except (httpx.ConnectError, httpx.TimeoutException):
            self._connected = False
        return self._connected

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- low-level RPC -------------------------------------------------------

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self._http:
            raise SMPError(-1, "Not connected")
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._req_id,
        }
        r = await self._http.post("/rpc", json=payload)
        if r.status_code == 204:
            return None
        body = r.json()
        if err := body.get("error"):
            raise SMPError(err["code"], err["message"])
        return body.get("result")

    # -- memory management ---------------------------------------------------

    async def update_file(self, file_path: str, content: str, change_type: str = "modified") -> dict[str, Any]:
        """Push a file change into the SMP graph."""
        return await self._rpc("smp/update", {
            "file_path": file_path,
            "content": content,
            "change_type": change_type,
        })

    async def batch_update(self, changes: list[dict[str, str]]) -> dict[str, Any]:
        return await self._rpc("smp/batch_update", {"changes": changes})

    async def reindex(self, scope: str = "full") -> dict[str, Any]:
        return await self._rpc("smp/reindex", {"scope": scope})

    # -- queries -------------------------------------------------------------

    async def navigate(self, query: str) -> dict[str, Any]:
        return await self._rpc("smp/navigate", {"query": query, "include_relationships": True})

    async def trace(self, start: str, depth: int = 3, direction: str = "outgoing") -> Any:
        return await self._rpc("smp/trace", {"start": start, "depth": depth, "direction": direction})

    async def get_context(self, file_path: str, scope: str = "edit", depth: int = 2) -> dict[str, Any]:
        return await self._rpc("smp/context", {"file_path": file_path, "scope": scope, "depth": depth})

    async def assess_impact(self, entity: str, change_type: str = "signature_change") -> dict[str, Any]:
        return await self._rpc("smp/impact", {"entity": entity, "change_type": change_type})

    async def locate(self, description: str, top_k: int = 10) -> Any:
        return await self._rpc("smp/locate", {"query": description, "top_k": top_k})

    async def search(self, query: str, top_k: int = 10) -> Any:
        return await self._rpc("smp/search", {"query": query, "top_k": top_k})

    async def flow(self, start: str, end: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"start": start}
        if end:
            params["end"] = end
        return await self._rpc("smp/flow", params)

    # -- safety --------------------------------------------------------------

    async def open_session(self, agent_id: str) -> dict[str, Any]:
        return await self._rpc("smp/session/open", {"agent_id": agent_id})

    async def close_session(self, session_id: str) -> dict[str, Any]:
        return await self._rpc("smp/session/close", {"session_id": session_id})

    async def guard_check(self, file_path: str, change_type: str) -> dict[str, Any]:
        return await self._rpc("smp/guard/check", {"file_path": file_path, "change_type": change_type})

    async def dryrun(self, file_path: str, content: str) -> dict[str, Any]:
        return await self._rpc("smp/dryrun", {"file_path": file_path, "content": content})
