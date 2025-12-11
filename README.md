# QA Agentic Application – LangGraph + FastAPI

This repository contains a Question-Answering Agentic System built using LangGraph, FastAPI, Redis (for checkpointing), MongoDB, and Qdrant.
It supports document ingestion, RAG pipelines, video/audio transcription, and S3-based processing.

---------------------------------------------------------------------------------------------------

## 1. System Requirements

Before setup, install the following on your machine:

Mandatory Tools

Miniconda – for environment management

Docker Desktop – for running Redis

Pytesseract, poppler

MongoDB (Local Installation)
 - MongoDB Compass (GUI client)- Select this checkbox while installation

---------------------------------------------------------------------------------------------------

## 2. Environment Setup
Step 1 — Create and Activate Conda Environment
conda create -n <env_name> python=3.10 -y
conda activate <env_name>

### Step 2 — Install Dependencies

Ensure your requirements.txt is present at the project root.

pip install -r requirements.txt

Install redis checkpoint for langgraph- pip install -U langgraph-checkpoint-redis

### Step 3 — Start Redis (Required for LangGraph Checkpointing)

Run the Redis Stack container:

docker run -d -p 6379:6379 --name redis-stack redis/redis-stack
Redis UI available at: http://localhost:8001

Redis server will be available at: localhost:6379

### Step 4 — Verify MongoDB is Running Locally

MongoDB should be accessible at the default URI:

mongodb://localhost:27017
You may open MongoDB Compass to verify.

-----------------------------------------------------------------------------------------------------

## 3. Running the FastAPI Application

Navigate to the folder where run.py exists:

python run.py


The server starts at:

http://localhost:8000

## 4. API Endpoints

Below are the core endpoints for the QA Agentic Application.

### 4.1 Document Processing Endpoint
POST /rag_doc_proccess

Used to upload and process documents (PDF, DOCX, TXT, Images).

### Example Request
http://localhost:8000/rag_doc_proccess

Body (JSON)
{
    "filepaths": ["<filepath>"],
    "user_id": "id",
    "session_id": "session1"
}

Description

This endpoint:
Extracts text from files
Performs chunking
Generates embeddings
Stores them in Qdrant/MongoDB
Returns processing metadata

### 4.2 S3 Video/Audio Processing Endpoint
POST /rag_s3_process

Use this to generate transcript → chunks → embeddings → storage.

### Request URL
http://localhost:8000/rag_s3_process

Body (JSON)
{
    "filepath": "video_file.mp4",
    "user_id": "user_id",
    "session_id": "session_id"
}


This pipeline performs:

Upload to S3
SToring metadata in dynamoDB
Whisper transcription
Chunking
Embedding
Storing in Qdrant
