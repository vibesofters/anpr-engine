FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS builder
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /build
RUN python -m pip install --no-cache-dir uv==0.12.5
COPY deploy/docker/requirements.cpu-arm64.txt deploy/docker/requirements.cpu-arm64.txt
RUN python -m venv /opt/venv && uv pip sync --python /opt/venv/bin/python --require-hashes deploy/docker/requirements.cpu-arm64.txt
COPY src/anpr_engine src/anpr_engine
COPY configs/recognition/cnn-transformer-dual-slot-tr-v1-foundation.yaml configs/recognition/cnn-transformer-dual-slot-tr-v1-foundation.yaml
COPY deploy/docker/create_python_notices.py deploy/docker/create_python_notices.py
RUN python deploy/docker/create_python_notices.py /opt/venv /build/licenses

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp XDG_CACHE_HOME=/tmp ANPR_RUNTIME_REPOSITORY_ROOT=/app ANPR_SERVICE_HOST=0.0.0.0 ANPR_SERVICE_PORT=8080 ANPR_PRIVATE_MODEL_BUNDLE=/models ANPR_INFERENCE_DEVICE=cpu
WORKDIR /app
COPY --from=builder --chown=10001:10001 /opt/venv/ /opt/venv/
COPY --from=builder --chown=10001:10001 /build/src/ /app/src/
COPY --from=builder --chown=10001:10001 /build/configs/ /app/configs/
COPY --from=builder --chown=10001:10001 /build/licenses/ /licenses/
COPY scripts/launch_private_inference_service.py /app/scripts/launch_private_inference_service.py
RUN mkdir -p /licenses/base && \
    dpkg-query -W -f='${Package}\t${Version}\n' > /licenses/base/debian-packages.tsv && \
    find /usr/share/doc -name copyright | while read -r file; do cp --parents "$file" /licenses/base/; done && \
    find /usr/share/doc -maxdepth 1 -type l -printf '%f -> %l\n' > /licenses/base/debian-doc-aliases.txt && \
    python -c 'import urllib.request,hashlib,pathlib; u="https://raw.githubusercontent.com/python/cpython/v3.12.14/LICENSE"; b=urllib.request.urlopen(u,timeout=20).read(); p=pathlib.Path("/licenses/base/PYTHON-LICENSE"); p.write_bytes(b); pathlib.Path("/licenses/base/python-license.sha256").write_text(hashlib.sha256(b).hexdigest()+"  PYTHON-LICENSE\n")'
USER 10001:10001
EXPOSE 8080
CMD ["python", "/app/scripts/launch_private_inference_service.py"]
