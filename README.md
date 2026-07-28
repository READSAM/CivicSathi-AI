# Civic Saathi AI Service

The Civic Saathi AI Service is a high-performance backend application built with FastAPI designed to classify civic complaints and intelligently route them to the appropriate government departments.

## 🚀 Features

* **Hybrid Classification Engine**: Implements a primary keyword-based matching strategy, utilizing a fallback mechanism to a Hugging Face zero-shot classification model (`facebook/bart-large-mnli`) for complex queries.


* **FastAPI Framework**: Provides lightweight, fast, and documented REST endpoints for issue analysis.


* **Dockerized Environment**: Fully containerized setup using a Python 3.11-slim image for easy and consistent deployment.


* **Uptime Maintenance**: Includes an automated scheduler script to ping hosted endpoints and prevent cloud instances (like Render) from spinning down.



---

## 🛠️ Prerequisites

Before you begin, ensure you have the following:

* **Python 3.11+** (if running locally without Docker)


* **Docker** (if running via containers)
* **Hugging Face Token**: An active API token is required to interface with the Hugging Face Router for transformer-based inference.



---

## ⚙️ Setup and Installation

### 1. Environment Variables

Create a `.env` file in the root directory and add your Hugging Face token:

```env
HF_TOKEN=your_hugging_face_access_token_here

```

### 2. Data Files

Ensure the following required CSV files are present in the root directory alongside your source code:

* `tag_departments.csv` - Contains mapping relationships between issue tags and department names.
* `tag_keywords.csv` - Contains keyword triggers and their associated issue tags.

### 3. Local Installation

To run the application locally on your machine:

1. Install the required dependencies:



```bash
pip install -r requirements.txt

```

2. Start the FastAPI server using Uvicorn:



```bash
uvicorn main:app --host 0.0.0.0 --port 8000

```

### 4. Docker Installation

To run the application using Docker:

1. Build the Docker image:



```bash
docker build -t civic-saathi-ai .

```

2. Run the container:



```bash
docker run -d -p 8000:8000 --env-file .env civic-saathi-ai

```

---

## 📡 API Endpoints

### Analyze Issue

**Endpoint:** `POST /analyze-issue`


**Description:** Analyzes a civic issue description and returns the predicted tag, department, and model confidence.

**Request Body:**

```json
{
  "description": "A tree fell over the road and is blocking traffic near the main intersection."
}

```

**Response:**

```json
{
  "tag": "road_hazard",
  "department": "Department of Transportation",
  "confidence": 1.0
}

```

### Health Check

**Endpoint:** `GET /health`


**Description:** Verifies the operational status of the service.

**Response:**

```json
{
  "status": "healthy",
  "service": "civic-saathi-ai"
}

```

### Root/Welcome

**Endpoint:** `GET /`


**Description:** Returns a welcome message confirming the service is live.

---

## 📂 Project Structure

* `main.py`: The FastAPI application setup, defining routing and payload validation schemas.


* `engine.py`: Contains the `CivicClassifier` class which manages data loading, keyword regex matching, and the external Hugging Face inference logic.


* `Dockerfile`: Configuration for building the Python 3.11-slim container.


* `requirements.txt`: Lists all Python package dependencies such as `fastapi`, `pandas`, `requests`, and `uvicorn`.


* `ping_script.py` (Implementation from Source 4): A utility script using the `schedule` and `requests` libraries to keep cloud deployments awake.
