FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your local data and source code
COPY tag_departments.csv .
COPY tag_keywords.csv .
COPY engine.py .
COPY main.py .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]