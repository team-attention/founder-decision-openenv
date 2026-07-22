FROM python:3.12.13-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir uv==0.11.26 && uv sync --frozen --no-dev
CMD ["uv", "run", "uvicorn", "yc_founder_decision_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
