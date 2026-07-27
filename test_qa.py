import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.services.prediction_service import predict_yield_list
from src.streamlit_app.utils.pdf_report import generate_pdf_report
from src.streamlit_app.utils.recommendations import generate_recommendations
from src.services.data_service import get_dataframe

test_cases = [
    {"state_name": "punjab", "district_name": "ludhiana", "crop": ["wheat"], "season": "rabi", "crop_year": 2025, "area": 5.0},
    {"state_name": "bihar", "district_name": "patna", "crop": ["rice"], "season": "kharif", "crop_year": 2028, "area": 3.0},
    {"state_name": "tamil nadu", "district_name": "coimbatore", "crop": ["sugarcane"], "season": "kharif", "crop_year": 2030, "area": 10.0},
    {"state_name": "maharashtra", "district_name": "nagpur", "crop": ["cotton(lint)"], "season": "kharif", "crop_year": 2027, "area": 4.0} # Used cotton(lint) as it's the typical db name
]

df = get_dataframe()

for idx, tc in enumerate(test_cases, 1):
    print(f"\n--- Running Test Case {idx}: {tc['state_name'].title()} | {tc['crop'][0].title()} | {tc['crop_year']} ---")
    
    # 1. Fallback Weather Logic Simulation
    # Simulate future year weather fetch returning empty, so we fallback
    hist_weather = df[(df['district_name'] == tc['district_name']) & (df['season'] == tc['season'])]
    if hist_weather.empty:
        hist_weather = df[df['district_name'] == tc['district_name']]
    if hist_weather.empty:
        hist_weather = df
        
    weather_summary = {
        "temperature_2m_mean": hist_weather["temperature_2m_mean"].mean(),
        "precipitation_sum": hist_weather["precipitation_sum"].mean(),
        "relative_humidity_2m_mean": hist_weather["relative_humidity_2m_mean"].mean(),
        "wind_speed_10m_mean": hist_weather["wind_speed_10m_mean"].mean()
    }
    
    tc.update(weather_summary)
    
    try:
        # 2. Prediction Execution
        predictions = predict_yield_list(tc)
        res = predictions[0]
        yield_val_ha = res["Yield"]
        final_features = res["Features"]
        
        # 3. Production Calculation
        total_production = yield_val_ha * tc["area"]
        print(f"Prediction: {yield_val_ha} T/Ha | Total Production: {total_production} T")
        
        # 4. Feature Importance updates
        shap_vals = res["ShapValues"]
        print(f"Feature Importance Count: {len(shap_vals)} (Expected 60)")
        
        # 5. Recommendations updates
        recs = generate_recommendations(tc["state_name"], tc["district_name"], tc["crop"][0], tc["season"], weather_summary, yield_val_ha)
        print(f"Generated {len(recs)} Recommendations")
        
        # 6. PDF Generation
        inputs_for_pdf = {
            "state": tc["state_name"], "district": tc["district_name"], "crop_year": tc["crop_year"], 
            "season": tc["season"], "area": tc["area"], "area_unit": "ha"
        }
        predictions_for_pdf = [{"Crop": tc["crop"][0], "Yield": yield_val_ha, "Production": total_production}]
        
        pdf_bytes = generate_pdf_report(inputs_for_pdf, predictions_for_pdf, weather_summary, recs)
        print(f"PDF Generated Successfully, size: {len(pdf_bytes)} bytes")
        
        # 7. CSV Generation
        history = [{
            "Timestamp": "2026-07-27",
            "State": tc["state_name"],
            "District": tc["district_name"],
            "Crop": tc["crop"][0],
            "Year": tc["crop_year"],
            "Season": tc["season"],
            "Area": tc["area"],
            "Unit": "ha",
            "Yield (T/Ha)": round(yield_val_ha, 3),
            "Production (T)": round(total_production, 3)
        }]
        csv_data = pd.DataFrame(history).to_csv(index=False)
        print(f"CSV Generated Successfully, row count: {len(csv_data.split(chr(10)))-1}")
        
    except Exception as e:
        print(f"Exception occurred: {e}")
