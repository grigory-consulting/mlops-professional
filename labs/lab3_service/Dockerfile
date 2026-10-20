# syntax=docker/dockerfile:1
# Lab 3: Image des Banking-Service (Encoder, Klassifikator, Vektorindex hinter FastAPI).
# Modelle und Index stecken NICHT im Image, sie werden beim Start eingehängt.
#
# Bauen (im Ordner labs/):
#   docker build -t mlops-lab3-service:0.1 lab3_service
# Starten:
#   docker run --rm -p 127.0.0.1:8000:8000 \
#     -v "$PWD/models:/data/models:ro" -v "$PWD/artifacts:/data/artifacts:ro" \
#     mlops-lab3-service:0.1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Kein Zugriff auf den Hugging Face Hub: Der Service lädt nur lokale Ordner.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    # Wurzel mit models/ und artifacts/ (siehe engine.Settings)
    LAB_BASE_DIR=/data

WORKDIR /srv

# torch zuerst und als CPU-Wheel: größte Schicht, ändert sich am seltensten.
# Vom Standard-Index käme auf x86-64 die CUDA-Fassung mit mehreren Gigabyte.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Der Code kommt zuletzt: Eine Änderung an app.py baut nur diese Schicht neu.
COPY *.py /srv/lab3_service/

# Nicht als root laufen.
RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "lab3_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
