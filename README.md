# Real Estate ML Forecaster

An end-to-end Machine Learning web application and API service that predicts real estate prices based on property features.

## 📁 Repository Structure

```
real-estate-ml-api/
├── backend/                  # Python FastAPI Backend & ML Model
│   ├── main.py               # FastAPI server endpoints
│   ├── model.py              # XGBoost training pipeline script
│   ├── xgb_model.pkl         # Trained XGBoost model
│   ├── AmesHousing.csv       # Training dataset
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Container configuration
│
├── frontend/                 # React.js + Vite Web Application
│   ├── src/                  # Components, styles, and UI logic
│   ├── public/               # Static assets
│   ├── package.json          # Node.js dependencies & scripts
│   └── vite.config.js        # Vite configuration
│
└── README.md
```

## 🚀 How to Run Locally

### 1. Start the Backend API (Python / FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
The FastAPI backend will start at `http://127.0.0.1:8000`.

### 2. Start the Frontend (React / Vite)
```bash
cd frontend
npm install
npm run dev
```
The frontend application will start at `http://localhost:5173`.

---

## 🐳 Docker Deployment (Backend)
To run the backend inside a Docker container:
```bash
cd backend
docker build -t real-estate-backend .
docker run -p 8000:8000 real-estate-backend
```
