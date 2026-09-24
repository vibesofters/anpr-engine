"""Guard the live public Compose candidate's private-model and quota boundary."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_public_compose_keeps_models_private_and_bounds_public_uploads() -> None:
    compose = yaml.safe_load(
        (ROOT / "deploy/docker/compose.public-candidate.yaml").read_text(encoding="utf-8")
    )
    web = compose["services"]["web"]
    inference = compose["services"]["inference"]

    assert set(compose["services"]) == {"web", "inference"}
    assert web["image"] == (
        "sha256:9a41b35a20bc17456d25e157c541483151b35e868414746d88061e907419f273"
    )
    assert inference["image"] == (
        "sha256:27d7192fba1ae159ce3443b1789729389940c642c5681eecafab489cbd4602e3"
    )
    assert web["environment"]["ANPR_PUBLIC_ORIGIN"] == ("https://anpr-engine.vibesofters.com")
    assert web["environment"]["ANPR_LOCAL_DIRECT_MODE"] == "0"
    assert web["environment"]["ANPR_INTERNAL_SERVICE_CREDENTIAL"] == (
        "${ANPR_INTERNAL_SERVICE_CREDENTIAL}"
    )
    assert inference["environment"]["ANPR_INTERNAL_SERVICE_CREDENTIAL"] == (
        "${ANPR_INTERNAL_SERVICE_CREDENTIAL}"
    )
    assert inference["environment"]["ANPR_APPLICATION_VERSION"] == ("${ANPR_APPLICATION_VERSION}")
    assert web["environment"]["ANPR_TRUSTED_PROXY_IP_HEADER"] == "x-real-ip"
    assert web["environment"]["ANPR_DAILY_QUOTA_PATH"] == "/quota/quota.sqlite"
    assert web["environment"]["ANPR_PROXY_PROOF_ENABLED"] == "0"
    assert "quota_data:/quota" in web["volumes"]
    assert "quota_data" in compose["volumes"]
    assert "ports" not in web and "ports" not in inference
    assert web["expose"] == ["3000"]
    assert inference["networks"] == ["private"]
    assert web["networks"] == ["front", "private"]
    assert inference["volumes"][0]["source"] == "${ANPR_PRIVATE_MODEL_BUNDLE_SOURCE}"
    assert compose["networks"]["private"]["internal"] is True
    assert inference["volumes"][0]["read_only"] is True
    assert inference["volumes"][0]["target"] == "/models"
    assert web["read_only"] is True and inference["read_only"] is True
    assert web["cpus"] == 0.25 and inference["cpus"] == 0.75
    assert web["mem_limit"] == "512m" and inference["mem_limit"] == "2048m"
    assert web["restart"] == "no" and inference["restart"] == "no"
