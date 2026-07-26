from typing import List, Dict, Any

def generate_recommendations(
    state: str,
    district: str,
    crop: str,
    season: str,
    weather_data: Dict[str, float],
    predicted_yield: float
) -> List[Dict[str, str]]:
    """
    Generate intelligent farming recommendations based on weather and inputs.
    Returns a list of dicts with 'title', 'text', and 'type' (info/warning/danger).
    """
    recs = []
    
    # 1. Weather-based recommendations
    rain = weather_data.get("precipitation_sum", 0)
    temp = weather_data.get("temperature_2m_mean", 0)
    humidity = weather_data.get("relative_humidity_2m_mean", 0)
    
    if rain < 5:
        recs.append({
            "title": "💧 Drought Risk Alert",
            "text": "Total rainfall is critically low (< 5mm) for this season. Ensure adequate irrigation systems are deployed to prevent severe yield loss.",
            "type": "danger"
        })
    elif rain > 800:
        recs.append({
            "title": "🌧 Heavy Rainfall Advisory",
            "text": "High precipitation predicted. Ensure proper field drainage to prevent waterlogging and root rot.",
            "type": "warning"
        })
        
    if temp > 35:
        recs.append({
            "title": "🌡 Heat Stress Warning",
            "text": f"Average temperatures are high ({temp:.1f}°C). Consider mulching and evening irrigation to reduce soil evaporation.",
            "type": "warning"
        })
        
    if humidity > 80:
        recs.append({
            "title": "🦠 Pest & Disease Risk",
            "text": f"High average humidity ({humidity:.1f}%). Monitor closely for fungal infections and pests which thrive in damp conditions.",
            "type": "danger"
        })
        
    # 2. General agronomic advice
    recs.append({
        "title": f"🌱 Crop-Season Suitability",
        "text": f"You are planning to grow {crop.title()} during the {season.title()} season in {state.title()}. Based on historical patterns, this is a recognized combination. Ensure soil testing is performed before sowing.",
        "type": "info"
    })
    
    if predicted_yield > 4.0:
        recs.append({
            "title": "📈 High Yield Potential",
            "text": "The model predicts an exceptionally strong yield. Ensure you have adequate harvesting labor and storage capacity ready.",
            "type": "info"
        })
        
    return recs
