# Vision-Based Student Behavior Analysis System

A comprehensive AI-powered web application designed to analyze classroom dynamics by detecting student behaviors, recognizing identities, and classifying facial emotions from video footage. This system aids educators and administrators in monitoring student engagement and improving teaching quality through data-driven insights.

## 🚀 Key Features

### 1\. 📊 Interactive Dashboard

  - **System Overview**: Real-time statistics on total students, processed sessions, and detected AI events.
  - **Visual Analytics**: Charts displaying behavior distribution and emotion trends across the system.
  - **Recent Activity**: Quick access to the latest processed class sessions.

### 2\. 👥 Student Management

  - **Profile Management**: Create, update, and manage detailed student profiles (ID, Name, Class, etc.).
  - **Face Registration**: Upload training photos to generate face embeddings.
  - **Vector Database**: Uses **FAISS** to store and retrieve student identities for automated attendance and tracking.

### 3\. 🎥 Class Session Analysis

  - **Session Metadata**: Manage class schedules, subjects, and teacher information.
  - **Video Upload**: Support for uploading classroom recordings.
  - **Asynchronous AI Processing**:
      - Videos are processed in the background without blocking the UI.
      - **Behavior Detection**: Detects actions like **Hand-raising, Reading, Writing** using **YOLOv8**.
      - **Face Recognition**: Identifies students within the video using **InsightFace**.
      - **Emotion Analysis**: Classifies 7 basic emotions (Happy, Sad, Neutral, etc.) using **ResNet18**.

### 4\. 📈 Comprehensive Reports

  - **Session Replay (Timeline)**: Watch class videos with synchronized bounding boxes showing detected behaviors, student identities, and emotions in real-time.
  - **Class-Level Summary**: Aggregate reports showing engagement trends over multiple sessions for a specific class and subject.
  - **Student Insights**: Detailed performance metrics for individual students, including attendance rate, dominant emotions, and activity scores.

### 5\. 🧪 Testing Zone

  - **Model Debugging**: Dedicated interface to test individual AI models.
  - **Upload & Test**: Upload static images to visualize YOLOv8 behavior detection and InsightFace emotion/identity recognition results instantly.

-----

## 🛠️ Technology Stack

### Backend & AI

  - **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+)
  - **Database**: MySQL (Relational Data), FAISS (Vector Embeddings)
  - **ORM**: SQLAlchemy
  - **Computer Vision**: OpenCV
  - **Deep Learning Models**:
      - **YOLOv8** (Ultralytics) for Object/Behavior Detection.
      - **InsightFace** (ArcFace) for Face Analysis.
      - **ResNet18** (PyTorch) for Facial Expression Recognition.

### Frontend

  - **Templating**: Jinja2 (Server-side rendering)
  - **Styling**: TailwindCSS + DaisyUI
  - **Interactivity**: Alpine.js
  - **Charts**: Chart.js

### Deployment

  - **Containerization**: Docker
  - **Server**: Uvicorn / Gunicorn
  - **Reverse Proxy**: Nginx (Recommended for production)

-----

## ⚙️ Installation & Setup

### Prerequisites

  - Python 3.11 or higher
  - MySQL Database Server
  - CUDA-capable GPU (Optional, but recommended for AI inference)

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd student_behavior_web
```

### 2\. Configure Environment

Create a `.env` file or update `core/config.py` with your database credentials:

```python
# core/config.py
DATABASE_HOST = "localhost"
DATABASE_PORT = 3306
DATABASE_NAME = "student_ai_db"
DATABASE_USERNAME = "root"
DATABASE_PASSWORD = "your_password"
```

### 3\. Install Dependencies

Recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4\. Prepare AI Models

Ensure the model weights are placed in the `models/` directory:

  - `models/yolov8_best.pt` (Trained YOLOv8 model)
  - `models/best_resnet18_sgd.ckpt` (Trained Emotion model)

### 5\. Run the Application

```bash
python server.py
# OR using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Access the web interface at: `http://localhost:8080`

-----

## 🐳 Running with Docker

The project is containerized for easy deployment.

1.  **Build and Run**:

    ```bash
    docker-compose up -d --build
    ```

2.  **Access**:
    The application will be running on port **8080**.

    *Note: For production, it is recommended to set up Nginx as a reverse proxy to handle static files and request forwarding (configuration provided in `nginx/` folder if applicable).*

-----

## 📂 Project Structure

```
student_behavior_web/
├── app/                # API Routers and View Controllers
│   ├── dashboard_api.py
│   ├── report_api.py
│   ├── session_api.py
│   ├── student_api.py
│   └── ...
├── core/               # Core logic and Configuration
│   ├── manager/        # Business logic (AI Pipeline, CRUD)
│   ├── ai_loader.py    # AI Model Initialization
│   └── database.py     # DB Connection
├── db/                 # Database Models & Vector DB
│   ├── models.py       # SQLAlchemy Models
│   └── vector_db.py    # FAISS Wrapper
├── templates/          # HTML Templates (Jinja2)
├── static/             # CSS, JS, Images
├── models/             # AI Model Weights (.pt, .ckpt)
├── main.py             # FastAPI Entry Point
├── server.py           # Server Runner
└── requirements.txt    # Python Dependencies
```

-----

## 📜 License

This project is part of a Master's Thesis submission. All rights reserved.