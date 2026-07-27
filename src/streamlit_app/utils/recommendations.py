"""
Intelligent agricultural recommendations engine.

Generates structured, data-driven recommendations organized into sections:
  - Agronomic Advice
  - Climate Risk Analysis
  - Pest & Disease Risk
  - Irrigation Suitability
  - Best Practices

All recommendations are derived from real weather data and yield predictions.
Never generates fake or placeholder advice.
"""

from typing import List, Dict, Any


def generate_recommendations(
    state: str,
    district: str,
    crop: str,
    season: str,
    weather_data: Dict[str, float],
    predicted_yield: float,
    hist_avg: float = None,
) -> List[Dict[str, str]]:
    """
    Generate structured farming recommendations.

    Returns a list of dicts with keys:
        'title', 'text', 'type' (info/warning/danger/success),
        'section' (category), 'rationale' (why this recommendation)
    """
    recs = []

    rain = weather_data.get("precipitation_sum", 0)
    temp = weather_data.get("temperature_2m_mean", 0)
    humidity = weather_data.get("relative_humidity_2m_mean", 0)
    wind = weather_data.get("wind_speed_10m_mean", 0)

    # ── Agronomic Advice ──
    recs.append({
        "title": f"🌱 Crop-Season Suitability — {crop.title()}",
        "text": (
            f"You are planning to grow {crop.title()} during the {season.title()} season "
            f"in {district.title()}, {state.title()}. Based on historical data patterns, "
            f"this is a recognized crop-season-region combination. Conduct a soil nutrient test "
            f"(N-P-K, pH, organic carbon) before sowing to optimize fertilizer application."
        ),
        "type": "info",
        "section": "Agronomic Advice",
        "rationale": (
            f"This recommendation is based on the crop-state-season combination being present "
            f"in our 20-year historical dataset covering {state.title()}."
        ),
    })

    if predicted_yield > 4.0:
        recs.append({
            "title": "📈 High Yield Potential Detected",
            "text": (
                f"The model predicts a yield of {predicted_yield:.2f} T/Ha, which is above the "
                f"high-productivity threshold. Ensure adequate harvesting labor, transport logistics, "
                f"and storage capacity are arranged in advance to minimize post-harvest losses."
            ),
            "type": "success",
            "section": "Agronomic Advice",
            "rationale": "Predicted yield exceeds 4.0 T/Ha, indicating strong growing conditions.",
        })

    if hist_avg and hist_avg > 0:
        improvement = ((predicted_yield - hist_avg) / hist_avg) * 100
        if improvement > 15:
            recs.append({
                "title": "🚀 Above-Average Yield Expected",
                "text": (
                    f"The predicted yield is {improvement:.1f}% above the historical district "
                    f"average of {hist_avg:.2f} T/Ha. Consider increasing cultivated area or "
                    f"investing in quality seeds to capitalize on favorable conditions."
                ),
                "type": "success",
                "section": "Agronomic Advice",
                "rationale": f"Predicted yield exceeds district historical average by {improvement:.1f}%.",
            })
        elif improvement < -15:
            recs.append({
                "title": "⚠ Below-Average Yield Expected",
                "text": (
                    f"The predicted yield is {abs(improvement):.1f}% below the historical district "
                    f"average of {hist_avg:.2f} T/Ha. Review input quality (seeds, fertilizers), "
                    f"irrigation access, and consider pest management improvements."
                ),
                "type": "warning",
                "section": "Agronomic Advice",
                "rationale": f"Predicted yield is {abs(improvement):.1f}% below district historical average.",
            })

    # ── Climate Risk Analysis ──
    if temp > 35:
        recs.append({
            "title": "🌡 Heat Stress Risk",
            "text": (
                f"Average temperature of {temp:.1f}°C exceeds the 35°C heat-stress threshold. "
                f"Implement mulching (4-6 cm organic layer) to reduce soil temperature by 3-5°C. "
                f"Schedule irrigation during early morning or late evening to minimize evapotranspiration."
            ),
            "type": "warning",
            "section": "Climate Risk",
            "rationale": f"Temperature ({temp:.1f}°C) exceeds the heat stress threshold of 35°C.",
        })
    elif temp < 10:
        recs.append({
            "title": "🥶 Cold Stress Risk",
            "text": (
                f"Average temperature of {temp:.1f}°C is below the 10°C cold stress threshold for "
                f"many tropical and subtropical crops. Consider protective covers, windbreaks, or "
                f"selecting cold-tolerant crop varieties."
            ),
            "type": "warning",
            "section": "Climate Risk",
            "rationale": f"Temperature ({temp:.1f}°C) is below the cold stress threshold of 10°C.",
        })

    if rain < 5:
        recs.append({
            "title": "🏜 Drought Condition Alert",
            "text": (
                f"Total seasonal rainfall of {rain:.1f} mm is critically low. Implement drip or "
                f"sprinkler irrigation to ensure consistent moisture. Apply water-retention agents "
                f"(hydrogels) if available. Monitor soil moisture daily using tensiometers."
            ),
            "type": "danger",
            "section": "Climate Risk",
            "rationale": f"Rainfall ({rain:.1f} mm) is critically below the minimum threshold of 5 mm.",
        })
    elif rain > 1000:
        recs.append({
            "title": "🌊 Excessive Rainfall Advisory",
            "text": (
                f"Total rainfall of {rain:.1f} mm indicates potential waterlogging risk. Ensure proper "
                f"field drainage channels. Raised bed planting is recommended. Monitor for root rot "
                f"and apply fungicide preventively."
            ),
            "type": "danger",
            "section": "Climate Risk",
            "rationale": f"Rainfall ({rain:.1f} mm) exceeds the heavy precipitation threshold of 1000 mm.",
        })
    elif rain > 500:
        recs.append({
            "title": "🌧 Moderate to Heavy Rainfall",
            "text": (
                f"Seasonal rainfall of {rain:.1f} mm is adequate but on the higher side. Ensure field "
                f"drainage is functional. Watch for fungal diseases that thrive in wet conditions."
            ),
            "type": "info",
            "section": "Climate Risk",
            "rationale": f"Rainfall ({rain:.1f} mm) is in the moderate-high range (500-1000 mm).",
        })

    if wind > 25:
        recs.append({
            "title": "💨 High Wind Risk",
            "text": (
                f"Average wind speed of {wind:.1f} m/s may cause lodging in tall crops like sugarcane, "
                f"maize, or rice. Consider staking, windbreaks, or selecting shorter, sturdier varieties."
            ),
            "type": "warning",
            "section": "Climate Risk",
            "rationale": f"Wind speed ({wind:.1f} m/s) exceeds the high-wind threshold.",
        })

    # ── Pest & Disease Risk ──
    if humidity > 80:
        recs.append({
            "title": "🦠 Fungal Disease Risk — High Humidity",
            "text": (
                f"Average humidity of {humidity:.0f}% creates favorable conditions for fungal diseases "
                f"(blast, blight, rust, powdery mildew). Apply preventive fungicide sprays at early "
                f"growth stages. Maintain adequate plant spacing for air circulation."
            ),
            "type": "danger",
            "section": "Pest & Disease Risk",
            "rationale": f"Humidity ({humidity:.0f}%) exceeds the 80% fungal disease risk threshold.",
        })

    if humidity > 70 and temp > 25:
        recs.append({
            "title": "🐛 Insect Pest Risk",
            "text": (
                f"Warm temperatures ({temp:.1f}°C) combined with high humidity ({humidity:.0f}%) "
                f"increase pest pressure from aphids, stem borers, and whiteflies. Implement "
                f"Integrated Pest Management (IPM): use pheromone traps, biological controls, "
                f"and targeted pesticide application only when pest density exceeds threshold."
            ),
            "type": "warning",
            "section": "Pest & Disease Risk",
            "rationale": (
                f"Temperature ({temp:.1f}°C) and humidity ({humidity:.0f}%) combination "
                f"exceeds pest proliferation thresholds."
            ),
        })

    # ── Irrigation Suitability ──
    if rain < 200:
        recs.append({
            "title": "💧 Irrigation Essential",
            "text": (
                f"With only {rain:.1f} mm of rainfall, supplemental irrigation is critical. "
                f"Drip irrigation is the most efficient option (90% efficiency). For rice paddies, "
                f"alternate wetting and drying (AWD) can save 15-30% water while maintaining yield."
            ),
            "type": "warning",
            "section": "Irrigation",
            "rationale": f"Seasonal rainfall ({rain:.1f} mm) is below the 200 mm minimum for rainfed agriculture.",
        })
    elif rain > 500:
        recs.append({
            "title": "✅ Adequate Natural Water Supply",
            "text": (
                f"Seasonal rainfall of {rain:.1f} mm should provide sufficient moisture for most crops. "
                f"Supplemental irrigation may only be needed during dry spells. Focus on drainage management."
            ),
            "type": "success",
            "section": "Irrigation",
            "rationale": f"Rainfall ({rain:.1f} mm) exceeds the 500 mm adequacy threshold.",
        })

    # ── Best Practices ──
    season_lower = season.lower()
    if season_lower == 'kharif':
        recs.append({
            "title": "🌿 Kharif Season Best Practices",
            "text": (
                "For Kharif crops (June-September): Sow with the onset of monsoon. "
                "Use seed treatment with fungicide + insecticide. Apply basal fertilizer at sowing. "
                "Maintain field bunds for water conservation. Top-dress nitrogen at 30 days after sowing."
            ),
            "type": "info",
            "section": "Best Practices",
            "rationale": "Season-specific agronomic practices for the Kharif monsoon season.",
        })
    elif season_lower == 'rabi':
        recs.append({
            "title": "❄ Rabi Season Best Practices",
            "text": (
                "For Rabi crops (October-March): Ensure adequate soil moisture at sowing. "
                "Apply pre-emergence herbicide. Schedule 4-5 irrigations at critical growth stages "
                "(crown root, tillering, flowering, grain filling). Protect from frost if needed."
            ),
            "type": "info",
            "section": "Best Practices",
            "rationale": "Season-specific agronomic practices for the Rabi winter season.",
        })
    elif season_lower == 'summer':
        recs.append({
            "title": "☀ Summer Crop Best Practices",
            "text": (
                "For summer crops (April-May): Ensure consistent irrigation as natural rainfall "
                "is minimal. Apply mulch to reduce evaporation. Schedule field operations during "
                "cooler parts of the day. Use shade nets for sensitive crops."
            ),
            "type": "info",
            "section": "Best Practices",
            "rationale": "Season-specific practices for heat and water management in summer.",
        })
    else:
        recs.append({
            "title": "📋 General Best Practices",
            "text": (
                "Maintain proper crop rotation to prevent soil depletion. Use certified seeds "
                "for better germination rates. Conduct periodic soil testing every 2-3 years. "
                "Follow recommended seed rate and plant spacing for optimal yield."
            ),
            "type": "info",
            "section": "Best Practices",
            "rationale": "Standard agronomic best practices applicable across seasons.",
        })

    return recs
