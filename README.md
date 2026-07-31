# Advanced Real Estate ML Forecaster

An end-to-end Machine Learning microservice and web application that predicts real estate prices based on historical housing data. 

## 🚀 Architecture
This project uses a decoupled architecture to separate the machine learning workloads from the user interface:
* **AI Engine:** XGBoost Regressor trained on the Ames Housing Dataset.
* **Backend API:** FastAPI (Python) for robust, type-checked data validation and model inference.
* **Frontend:** React.js (Vite) for a seamless, interactive user dashboard.
* **Deployment:** Fully containerized using Docker for easy scaling.

## 🧠 Machine Learning Pipeline
* **Data Processing:** Cleaned and engineered features using Pandas (focusing on Living Area, Bedrooms, Year Built, and Bathrooms).
* **Model:** XGBoost was chosen for its high performance on tabular data and robust handling of non-linear relationships.
* **Serialization:** The trained model is serialized using `joblib` for rapid loading into the FastAPI microservice.

## 💻 How to Run Locally
1. Clone the repository.
2. Start the FastAPI backend: `uvicorn main:app --reload` (runs on port 8000).
3. Start the React frontend: `npm run dev` (runs on port 5173).
