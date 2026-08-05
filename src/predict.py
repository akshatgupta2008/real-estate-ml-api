"""
Real Estate ML Pipeline: Interactive Property Predictor CLI Tool
-----------------------------------------------------------------
Allows users to input custom home characteristics and get real-time price predictions.
"""

import sys
import argparse
from pathlib import Path

# Sanitize sys.path
sys.path = [p for p in sys.path if 'google-cloud-sdk' not in p]
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
from src.data_processing import preprocess_input_dict, FEATURE_COLUMNS
from src.model import MODEL_SAVE_PATH, train_and_serialize_model

def load_or_train_model():
    if not MODEL_SAVE_PATH.exists():
        print("[INFO] Serialized model not found. Training model now...")
        train_and_serialize_model()
    return joblib.load(MODEL_SAVE_PATH)

def predict_property_value(property_dict: dict) -> float:
    model = load_or_train_model()
    df_input = preprocess_input_dict(property_dict)
    prediction = model.predict(df_input)[0]
    return float(prediction)

def main():
    parser = argparse.ArgumentParser(description="Real Estate Valuation Predictor")
    parser.add_argument("--sqft", type=float, default=None, help="Above ground living area (sq ft)")
    parser.add_argument("--qual", type=int, default=None, help="Overall quality rating (1-10)")
    parser.add_argument("--year", type=int, default=None, help="Year built")
    parser.add_argument("--garage", type=int, default=None, help="Garage vehicle capacity (cars)")
    parser.add_argument("--bsmt", type=float, default=None, help="Total basement area (sq ft)")
    parser.add_argument("--baths", type=int, default=None, help="Full bathrooms")
    parser.add_argument("--sample", action="store_true", help="Run with sample luxury home data")
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("         REAL ESTATE AUTOMATED VALUATION MODEL (AVM) CLI        ")
    print("=" * 65)
    
    if args.sample or args.sqft is not None:
        property_input = {
            'Gr Liv Area': args.sqft if args.sqft else 2200,
            'Overall Qual': args.qual if args.qual else 8,
            'Year Built': args.year if args.year else 2010,
            'Garage Cars': args.garage if args.garage else 2,
            'Total_Bsmt_SF': args.bsmt if args.bsmt else 1200,
            'Full Bath': args.baths if args.baths else 2,
        }
    else:
        print("\nInteractive Property Input Mode (press Enter for defaults):")
        try:
            sqft_str = input(" -> Living Area sq ft [Default 1,800]: ").strip()
            sqft = float(sqft_str) if sqft_str else 1800.0
            
            qual_str = input(" -> Overall Quality (1-10) [Default 7]: ").strip()
            qual = int(qual_str) if qual_str else 7
            
            year_str = input(" -> Year Built [Default 2005]: ").strip()
            year = int(year_str) if year_str else 2005
            
            garage_str = input(" -> Garage Capacity (cars) [Default 2]: ").strip()
            garage = int(garage_str) if garage_str else 2
            
            bsmt_str = input(" -> Basement sq ft [Default 1,000]: ").strip()
            bsmt = float(bsmt_str) if bsmt_str else 1000.0
            
            baths_str = input(" -> Full Bathrooms [Default 2]: ").strip()
            baths = int(baths_str) if baths_str else 2
            
            property_input = {
                'Gr Liv Area': sqft,
                'Overall Qual': qual,
                'Year Built': year,
                'Garage Cars': garage,
                'Total_Bsmt_SF': bsmt,
                'Full Bath': baths
            }
        except Exception as e:
            print(f"[NOTE] Input parsing fallback to standard property. Error: {e}")
            property_input = {'Gr Liv Area': 1800, 'Overall Qual': 7, 'Year Built': 2005, 'Garage Cars': 2, 'Total_Bsmt_SF': 1000, 'Full Bath': 2}

    price = predict_property_value(property_input)
    
    print("\n--- VALUATION SUMMARY ---")
    print(f" Property Specs  : {property_input.get('Gr Liv Area', 1800):,.0f} sq ft | {property_input.get('Overall Qual', 7)} Qual | Built {property_input.get('Year Built', 2005)}")
    print(f" Garage & Bsmt   : {property_input.get('Garage Cars', 2)} Cars | {property_input.get('Total_Bsmt_SF', 1000):,.0f} sq ft Bsmt")
    print(f" ESTIMATED VALUE : ${price:,.2f}")
    print("-------------------------\n")

if __name__ == "__main__":
    main()
