#!/usr/bin/env python3
"""Launch legacy owner/development review tooling on loopback only.

This persistent multipart/batch browser is not the planned public application
and must never be used as a container or public deployment entry point.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from anpr_engine.integration.browser import AnprV1Application, serve

LOCAL_ONLY_NOT_FOR_PUBLIC_DEPLOYMENT = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local owner/development review tool; not for public deployment."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model-bundle",
        type=Path,
        required=True,
        help=(
            "Owner-only directory containing detection.pt, recognition.pt, and bundle-manifest.json"
        ),
    )
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-open-browser", action="store_true")
    arguments = parser.parse_args()
    application = AnprV1Application(
        arguments.repository,
        model_bundle=arguments.model_bundle,
        device=arguments.device,
    )
    url = f"http://127.0.0.1:{arguments.port}"
    print("LOCAL-ONLY legacy review tool; persistent batches are not public-service behavior.")
    print(f"ANPR V1 browser ready: {url}")
    print(application.health())
    if not arguments.no_open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        serve(application, port=arguments.port)
    except KeyboardInterrupt:
        print("\nANPR V1 project-models-only browser stopped.")


if __name__ == "__main__":
    main()
