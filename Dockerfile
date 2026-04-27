FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY artifacts /app/artifacts
COPY data /app/data

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "tscp_slm.server:app", "--host", "0.0.0.0", "--port", "8000"]
