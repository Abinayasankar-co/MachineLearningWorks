FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PDM_CHECK_UPDATE=false \
    PDM_USE_VENV=true \
    PATH="/opt/pdm/bin:/workspace/.venv/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgomp1 \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir pdm \
    && pdm --version

WORKDIR /workspace

COPY pyproject.toml .python-version ./
COPY scripts ./scripts
COPY CropIdentification ./CropIdentification
COPY retailerproductidentification ./retailerproductidentification
COPY README.md ./

RUN pdm use -f 3.12 \
    && pdm install \
    && pdm run check-env

EXPOSE 8888

CMD ["pdm", "run", "jupyter"]
