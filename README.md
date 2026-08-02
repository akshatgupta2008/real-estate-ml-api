# Real Estate ML Forecaster & Market Analytics Dashboard

An end-to-end Machine Learning web application and analytics platform that predicts real estate values and delivers interactive market insights based on property features.

## 📁 Repository Structure

```
real-estate-ml-api/
├── backend/                  # Python FastAPI Backend & ML Model
│   ├── main.py               # FastAPI server endpoints & analytics APIs
│   ├── model.py              # XGBoost training pipeline script
│   ├── xgb_model.pkl         # Trained XGBoost regression model
│   ├── AmesHousing.csv       # Dataset for ML training & market analytics
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Container configuration
│
├── frontend/                 # React.js + Vite Web Application
│   ├── src/                  # Components, styles, Recharts graphs & icons
│   ├── public/               # Static assets
│   ├── package.json          # Node.js dependencies & scripts
│   └── vite.config.js        # Vite configuration
│
├── docs/
│   └── app_preview.png       # Application UI screenshot
├── package.json              # Root single-command starter configuration
└── README.md
```

## 🖼️ Application Interface & Features

![Application Preview](docs/app_preview.png)

### Key Features & Dashboard Modules

- **AI Valuation Estimator Tab**:
  - **Live Input Parameters**: Living Area (`Gr_Liv_Area`), Bedrooms (`Bedroom_AbvGr`), Year Built (`Year_Built`), Full Bathrooms (`Full_Bath`).
  - **Instant Valuation Engine**: Real-time evaluation powered by XGBoost regression.
  - **Smart Metrics**: Calculated Price per Sq Ft and Market Rank Percentile.
  - **Dynamic Snapshot Card**: Real-time property parameter preview.

- **Market Analytics Dashboard Tab**:
  - **Dataset KPI Bar**: Overall dataset sample size count, average market sale price, and average price per sq ft.
  - **Historical Trajectory (Area Chart)**: Interactive decade-by-decade home price trends (1950s–2010s).
  - **Property Brackets (Bar Chart)**: Average sale price and price/sq ft breakdown by living area size tiers.
  - **XGBoost Model Feature Impact**: Visual weight breakdown of input features driving home valuations.

---

## ⚡ API Endpoints (FastAPI)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/predict` | `POST` | Generates estimated home value, $/sq ft, and market percentile rank. |
| `/analytics/kpis` | `GET` | Returns overall dataset metrics (average price, avg $/sq ft, sample count). |
| `/analytics/trends` | `GET` | Returns historical price and size trends grouped by construction decade. |
| `/analytics/price-vs-sqft` | `GET` | Returns price averages grouped by living area size brackets. |
| `/analytics/feature-importance` | `GET` | Returns XGBoost model feature weights (% relative importance). |

---

## 🚀 How to Run Locally

### Option 1: Single Command (Recommended)

From the project root directory (`real-estate-ml-api`):
```bash
npm start
```
This will launch both the FastAPI backend (`http://127.0.0.1:8000`) and the Vite React frontend (`http://localhost:5173`) simultaneously in one terminal window with colorized logs!

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
The FastAPI backend will start at `http://127.0.0.1:8000` (Interactive API documentation at `http://127.0.0.1:8000/docs`).

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
