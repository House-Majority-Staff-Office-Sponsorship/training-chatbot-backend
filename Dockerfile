FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastapi/ .

ENV PORT=3001

# Don't run as root
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 appuser
USER appuser

EXPOSE 3001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]
