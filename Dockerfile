FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

COPY lib/app/ ./app/

EXPOSE 9001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1"]
