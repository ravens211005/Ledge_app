FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch (matches the 2.5.1 you use locally, without the CUDA libraries)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY config.py inference.py hazard_analysis.py visualize.py app.py ./
COPY depth_anything_v2 ./depth_anything_v2
COPY models ./models

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]