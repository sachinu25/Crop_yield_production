"""
PDF report generator — premium layout with executive dashboard metrics.

Generates a professional PDF report with:
  - Farm profile
  - AI prediction results with historical comparisons
  - Prediction metadata (ID, timestamp, model version)
  - Weather profile
  - Structured recommendations
"""

import io
import uuid
from datetime import datetime
from typing import Dict, Any, List

from fpdf import FPDF


class CropReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 18)
        self.set_text_color(5, 150, 105)  # Emerald
        self.cell(0, 10, 'Agri AI - Crop Yield Intelligence Report', border=0, ln=1, align='C')

        self.set_font('helvetica', '', 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Model: CatBoost v2.0.0', border=0, ln=1, align='C')

        self.set_draw_color(226, 232, 240)
        self.line(10, 28, 200, 28)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Agri AI Platform | Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, number: int, title: str):
        self.set_font('helvetica', 'B', 13)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, f'{number}. {title}', ln=1)
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def key_value(self, key: str, value: str):
        self.set_font('helvetica', '', 10)
        self.set_text_color(100, 116, 139)
        self.cell(80, 7, key, ln=0)
        self.set_text_color(15, 23, 42)
        self.set_font('helvetica', 'B', 10)
        self.cell(0, 7, value, ln=1)


def generate_pdf_report(
    inputs: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    weather: Dict[str, float],
    recommendations: List[Dict[str, str]],
    exec_metrics: Dict[str, Any] = None,
) -> bytes:
    """Generate a premium PDF report and return as bytes."""

    pdf = CropReportPDF()
    pdf.add_page()

    # 1. Farm Profile
    pdf.section_title(1, "Farm Profile")
    pdf.key_value("State:", inputs.get('state', '').title())
    pdf.key_value("District:", inputs.get('district', '').title())
    pdf.key_value("Season:", inputs.get('season', '').title())
    pdf.key_value("Crop Year:", str(inputs.get('crop_year', '')))
    area = inputs.get('area', 0)
    unit = inputs.get('area_unit', 'Hectares')
    pdf.key_value("Cultivated Area:", f"{area:,.2f} {unit}")
    pdf.ln(4)

    # 2. Prediction Results
    pdf.section_title(2, "AI Prediction Results")
    for pred in predictions:
        crop = pred.get('Crop', '').title()
        yield_val = pred.get('Yield', 0)
        production = pred.get('Production', 0)

        pdf.set_font('helvetica', 'B', 11)
        pdf.set_text_color(5, 150, 105)
        pdf.cell(0, 8, f"Crop: {crop}", ln=1)

        pdf.set_font('helvetica', '', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.key_value("Predicted Yield:", f"{yield_val:,.3f} Tonnes / Hectare")
        if production:
            pdf.key_value("Estimated Production:", f"{production:,.3f} Tonnes")
    pdf.ln(2)

    # 3. Executive Metrics (if available)
    if exec_metrics:
        pdf.section_title(3, "Executive Dashboard Metrics")
        metric_items = [
            ("Historical District Average", f"{exec_metrics.get('district_avg', 0):.3f} T/Ha"),
            ("Historical State Average", f"{exec_metrics.get('state_avg', 0):.3f} T/Ha"),
            ("National Average", f"{exec_metrics.get('national_avg', 0):.3f} T/Ha"),
            ("Difference vs Historical", f"{exec_metrics.get('diff_vs_hist', 0):+.3f} T/Ha"),
            ("Yield Improvement", f"{exec_metrics.get('improvement_pct', 0):+.1f}%"),
            ("Confidence Level", exec_metrics.get('confidence', 'N/A')),
            ("Prediction ID", exec_metrics.get('prediction_id', 'N/A')),
            ("Model Used", "CatBoost Regressor v2.0.0"),
            ("Weather Source", exec_metrics.get('weather_source', 'N/A')),
            ("Inference Time", f"{exec_metrics.get('inference_ms', 0):.0f} ms"),
        ]
        for key, val in metric_items:
            pdf.key_value(f"{key}:", val)
        pdf.ln(2)
        section_offset = 4
    else:
        section_offset = 3

    # Weather Profile
    pdf.section_title(section_offset, "Seasonal Weather Profile")
    pdf.key_value("Average Temperature:", f"{weather.get('temperature_2m_mean', 0):.2f} C")
    pdf.key_value("Total Rainfall:", f"{weather.get('precipitation_sum', 0):.2f} mm")
    pdf.key_value("Average Humidity:", f"{weather.get('relative_humidity_2m_mean', 0):.2f} %")
    pdf.key_value("Average Wind Speed:", f"{weather.get('wind_speed_10m_mean', 0):.2f} m/s")
    pdf.ln(4)

    # Recommendations
    pdf.section_title(section_offset + 1, "Agronomic Recommendations")
    current_section = None
    for rec in recommendations:
        section = rec.get('section', '')
        if section and section != current_section:
            current_section = section
            pdf.set_font('helvetica', 'B', 10)
            pdf.set_text_color(5, 150, 105)
            clean_section = ''.join(c for c in section if ord(c) < 128)
            pdf.cell(0, 8, clean_section, ln=1)

        pdf.set_font('helvetica', 'B', 10)
        pdf.set_text_color(15, 23, 42)
        clean_title = ''.join(c for c in rec.get('title', '') if ord(c) < 128).strip()
        pdf.cell(0, 7, clean_title, ln=1)

        pdf.set_font('helvetica', '', 9)
        pdf.set_text_color(71, 85, 105)
        clean_text = ''.join(c for c in rec.get('text', '') if ord(c) < 128)
        pdf.multi_cell(0, 5, clean_text)

        rationale = rec.get('rationale', '')
        if rationale:
            pdf.set_font('helvetica', 'I', 8)
            pdf.set_text_color(148, 163, 184)
            clean_rationale = ''.join(c for c in rationale if ord(c) < 128)
            pdf.multi_cell(0, 5, f"Rationale: {clean_rationale}")

        pdf.ln(2)

    return pdf.output(dest='S')
