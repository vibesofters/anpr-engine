"""Private, no-retention ASGI boundary for the frozen ANPR pipeline."""

from anpr_engine.inference_service.app import create_app

__all__ = ["create_app"]
