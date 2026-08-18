FROM python:3.11-slim

WORKDIR /app

# libGL/glib нужны OpenCV (cv2 тянет их даже без GUI-кода в проде)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=120 --retries 5 --no-cache-dir \
    --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
