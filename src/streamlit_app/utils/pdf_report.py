import io
from datetime import datetime
from typing import Dict, Any, List
from fpdf import FPDF

class CropReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 18)
        self.set_text_color(6, 78, 59) # Emerald 900
        self.cell(0, 10, 'Agri AI - Intelligent Crop Yield Report', border=0, ln=1, align='C')
        
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', border=0, ln=1, align='C')
        
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(
    inputs: Dict[str, Any], 
    predictions: List[Dict[str, Any]], 
    weather: Dict[str, float],
    recommendations: List[Dict[str, str]]
) -> bytes:
    """Generate a PDF report and return as bytes."""
    
    pdf = CropReportPDF()
    pdf.add_page()
    
    # 1. Input Summary
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42) # Slate 900
    pdf.cell(0, 10, '1. Farm Profile', ln=1)
    
    pdf.set_font('helvetica', '', 11)
    pdf.set_text_color(51, 65, 85)
    
    profile_data = [
        f"State: {inputs['state'].title()}",
        f"District: {inputs['district'].title()}",
        f"Season: {inputs['season'].title()}",
        f"Crop Year: {inputs['crop_year']}",
        f"Cultivated Area: {inputs['area']:,.2f} {inputs.get('area_unit', 'Hectares')}"
    ]
    for line in profile_data:
        pdf.cell(0, 8, line, ln=1)
    pdf.ln(5)
    
    # 2. Prediction Results
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '2. AI Prediction Results', ln=1)
    
    for pred in predictions:
        pdf.set_font('helvetica', 'B', 12)
        pdf.set_text_color(16, 185, 129) # Emerald 500
        crop = pred['Crop'].title()
        yield_val = pred['Yield']
        
        pdf.cell(0, 8, f"Crop: {crop}", ln=1)
        pdf.set_font('helvetica', '', 11)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(0, 8, f"- Predicted Yield: {yield_val:,.2f} Tonnes / Hectare", ln=1)
        
        if 'Production' in pred:
            pdf.cell(0, 8, f"- Estimated Total Production: {pred['Production']:,.2f} Tonnes", ln=1)
        pdf.ln(5)
        
    # 3. Weather Profile
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '3. Seasonal Weather Profile', ln=1)
    
    pdf.set_font('helvetica', '', 11)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 8, f"Average Temperature: {weather.get('temperature_2m_mean', 0):.2f} C", ln=1)
    pdf.cell(0, 8, f"Total Rainfall: {weather.get('precipitation_sum', 0):.2f} mm", ln=1)
    pdf.cell(0, 8, f"Average Humidity: {weather.get('relative_humidity_2m_mean', 0):.2f} %", ln=1)
    pdf.cell(0, 8, f"Average Wind Speed: {weather.get('wind_speed_10m_mean', 0):.2f} m/s", ln=1)
    pdf.ln(5)
    
    # 4. Intelligent Recommendations
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '4. Agronomic Recommendations', ln=1)
    
    for rec in recommendations:
        pdf.set_font('helvetica', 'B', 11)
        # Strip emoji from title for FPDF basic font compatibility
        clean_title = ''.join(c for c in rec['title'] if ord(c) < 128).strip()
        pdf.cell(0, 8, clean_title, ln=1)
        
        pdf.set_font('helvetica', '', 10)
        clean_text = ''.join(c for c in rec['text'] if ord(c) < 128)
        pdf.multi_cell(0, 6, clean_text)
        pdf.ln(3)

    return pdf.output(dest='S')
