#!/usr/bin/env python3
"""Launch the private, internal-network-only Phase 2B service."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("ANPR_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("ANPR_SERVICE_PORT", "8080"))
    uvicorn.run(
        "anpr_engine.inference_service.app:app",
        host=host,
        port=port,
        workers=1,
        access_log=False,
        server_header=False,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
