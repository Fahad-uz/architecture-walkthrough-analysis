FROM node:22-bookworm-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM node:22-bookworm-slim AS glb-tools

WORKDIR /opt/glb
COPY tools/glb/package.json tools/glb/package-lock.json ./
# Keep optional dependencies: sharp selects its native glibc/libvips package
# for the build architecture during npm ci.
RUN npm ci --omit=dev --include=optional
COPY tools/glb/optimize.mjs ./
RUN node --check optimize.mjs \
    && node --input-type=module -e \
      "import sharp from 'sharp'; const image = await sharp({create:{width:1,height:1,channels:4,background:{r:0,g:0,b:0,alpha:1}}}).webp().toBuffer(); if (!image.length) process.exit(1)"


FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    blender \
    ffmpeg \
    libatomic1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend-build /frontend/dist /app/frontend/dist
# The optimizer invokes Node directly and resolves all JS/native modules from
# /app/tools/glb. Both stages are Debian Bookworm, so sharp's glibc binary and
# the copied Node 22 runtime remain ABI-compatible in the Python image.
COPY --from=glb-tools /usr/local/bin/node /usr/local/bin/node
COPY --from=glb-tools /opt/glb /app/tools/glb
RUN node -e "if (!process.versions.node.startsWith('22.')) process.exit(1)" \
    && cd /app/tools/glb \
    && node --check optimize.mjs \
    && node --input-type=module -e \
      "import sharp from 'sharp'; import {NodeIO} from '@gltf-transform/core'; const image = await sharp({create:{width:1,height:1,channels:4,background:{r:0,g:0,b:0,alpha:1}}}).webp().toBuffer(); if (!image.length || !NodeIO) process.exit(1)"
RUN pip install --no-cache-dir --no-deps -e .

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "architecture_walkthrough.ui:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
