"""Frozen end-to-end ANPR integration surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anpr_engine.integration.anpr_v1 import AnprV1Pipeline, FrozenBundle

if TYPE_CHECKING:
    from anpr_engine.integration.review_store import BatchReviewStore


def __getattr__(name: str) -> Any:
    if name == "BatchReviewStore":
        from anpr_engine.integration.review_store import BatchReviewStore

        return BatchReviewStore
    raise AttributeError(name)


__all__ = ["AnprV1Pipeline", "BatchReviewStore", "FrozenBundle"]
