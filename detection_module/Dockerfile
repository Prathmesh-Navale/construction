FROM python:3.10-slim

WORKDIR /app

# FORCE torch downgrade (kills 2.6 completely)
RUN pip uninstall -y torch torchvision torchaudio || true

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
