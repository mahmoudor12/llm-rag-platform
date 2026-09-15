FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Kein build-essential, kein curl — alle Wheels sind vorcompiliert
# (spart ~400 MB Image-Größe und ~35 Sekunden Build-Zeit)

COPY requirements.txt .

# WICHTIG: PyTorch CPU-only zuerst installieren.
# Ohne diesen Schritt zieht pip die CUDA-Version mit ~2,5 GB NVIDIA-Paketen,
# die auf Intel-Hardware nutzlos sind.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

# Restliche Dependencies (nutzt das bereits installierte CPU-torch)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]