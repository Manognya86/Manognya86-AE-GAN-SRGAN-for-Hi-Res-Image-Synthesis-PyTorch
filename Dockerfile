FROM python:3.11-slim
WORKDIR /app
COPY results/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY results .
CMD ["python", "deploy.py"]