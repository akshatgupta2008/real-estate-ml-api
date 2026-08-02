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

### Option 1: Single Command (Both Frontend & Backend)

From the project root directory (`real-estate-ml-api`):
```bash
npm start
```
This will launch both the FastAPI backend (`http://127.0.0.1:8000`) and the Vite React frontend (`http://localhost:5173`) simultaneously in one terminal window!

---

### Option 2: Run in Separate Terminals

#### 1. Start the Backend API (Python / FastAPI)

**Windows (PowerShell):**
```powershell
# Activate virtual environment
.\env\Scripts\Activate.ps1

# Navigate to backend and start server
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**macOS / Linux:**
```bash
# Activate virtual environment
source env/bin/activate

# Navigate to backend and start server
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
The FastAPI backend will start at `http://127.0.0.1:8000` (Interactive API docs at `http://127.0.0.1:8000/docs`).

#### 2. Start the Frontend (React / Vite)
In a **second terminal window**:
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
