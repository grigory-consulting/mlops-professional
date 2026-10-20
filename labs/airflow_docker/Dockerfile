# syntax=docker/dockerfile:1
# Airflow für die Pipeline aus Lab 2 (dags/banking_pipeline.py), auch unter Windows (Airflow läuft dort nicht nativ).
# Airflow und die Stufen-Skripte bekommen je eine eigene Python-Umgebung, wie auf dem Host
# (.venv-airflow und .venv): Airflow bringt strenge Versionsvorgaben mit.
# Code, Daten und Modelle stecken NICHT im Image, compose.yaml hängt den Ordner labs/ ein.
FROM apache/airflow:3.3.2-python3.11

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Lab-Umgebung für die Stufen-Skripte; der DAG ruft sie über LAB_PYTHON auf.
# torch als CPU-Wheel: Vom Standard-Index käme auf x86-64 die CUDA-Fassung mit mehreren Gigabyte.
RUN python -m venv /home/airflow/lab-venv \
 && /home/airflow/lab-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY requirements.txt /tmp/lab-requirements.txt
RUN /home/airflow/lab-venv/bin/pip install -r /tmp/lab-requirements.txt
