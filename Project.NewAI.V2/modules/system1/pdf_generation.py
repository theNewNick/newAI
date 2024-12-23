import io
import logging
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

logger = logging.getLogger(__name__)

def generate_pdf_report(
    financials, ratios, cagr_values, sentiment_results,
    plots, recommendation, intrinsic_value_per_share,
    stock_price, weighted_total_score, weights, factor_scores,
    company_summary, industry_summary, risks_summary,
    company_logo_path=None
):
    logger.debug('Starting PDF report generation.')
    pdf_output = io.BytesIO()
    doc = SimpleDocTemplate(pdf_output, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    styles['Heading1'].fontSize = 18
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceAfter = 12
    styles['Heading2'].fontSize = 14
    styles['Heading2'].leading = 18
    styles['Heading2'].spaceAfter = 10
    styles['Normal'].fontSize = 12
    styles['Normal'].leading = 14
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 12
    centered_style = ParagraphStyle('Centered', alignment=TA_CENTER, fontSize=12)

    # Title Page
    elements.append(Paragraph("Financial Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    if company_logo_path:
        logo = Image(company_logo_path, width=200, height=100)
        elements.append(logo)
        elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Company: {financials.get('company_name', 'N/A')}", centered_style))
    elements.append(Paragraph(f"Report Date: {pd.Timestamp('today').strftime('%Y-%m-%d')}", centered_style))
    elements.append(PageBreak())

    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading1']))
    summary_text = f"""
    This report provides a comprehensive financial analysis of {financials.get('company_name', 'the company')}. The analysis includes Discounted Cash Flow (DCF), ratio analysis, time series analysis, sentiment analysis from various reports, and data visualizations. The final recommendation based on the weighted factors is: <b>{recommendation}</b>.
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Company Summary
    elements.append(Paragraph("Company Summary", styles['Heading1']))
    elements.append(Paragraph(company_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Industry Summary
    elements.append(Paragraph("Industry Summary", styles['Heading1']))
    elements.append(Paragraph(industry_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Risk Considerations
    elements.append(Paragraph("Risk Considerations", styles['Heading1']))
    elements.append(Paragraph(risks_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Financial Analysis
    elements.append(Paragraph("Financial Analysis", styles['Heading1']))

    # DCF Analysis
    elements.append(Paragraph("Discounted Cash Flow (DCF) Analysis", styles['Heading2']))
    dcf_text = f"""
    - Intrinsic Value per Share: ${intrinsic_value_per_share:.2f}<br/>
    - Current Stock Price: ${stock_price}<br/>
    - Factor 1 Score (DCF Analysis): {factor_scores['factor1_score']}
    """
    elements.append(Paragraph(dcf_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Ratio Analysis
    elements.append(Paragraph("Ratio Analysis", styles['Heading2']))
    ratio_data = [
        ["Ratio", "Value"],
        ["Debt-to-Equity Ratio", f"{ratios['Debt-to-Equity Ratio']:.2f}" if ratios['Debt-to-Equity Ratio'] is not None else "N/A"],
        ["Current Ratio", f"{ratios['Current Ratio']:.2f}" if ratios['Current Ratio'] is not None else "N/A"],
        ["P/E Ratio", f"{ratios['P/E Ratio']:.2f}" if ratios['P/E Ratio'] is not None else "N/A"],
        ["P/B Ratio", f"{ratios['P/B Ratio']:.2f}" if ratios['P/B Ratio'] is not None else "N/A"],
        ["Factor 2 Score (Ratio Analysis)", f"{factor_scores['factor2_score']}"],
    ]
    ratio_table = Table(ratio_data, hAlign='LEFT')
    ratio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(ratio_table)
    elements.append(Spacer(1, 12))

    # Time Series Analysis
    elements.append(Paragraph("Time Series Analysis", styles['Heading2']))
    cagr_data = [
        ["Metric", "CAGR"],
        ["Revenue CAGR", f"{cagr_values['revenue_cagr']:.2%}" if cagr_values['revenue_cagr'] is not None else "N/A"],
        ["Net Income CAGR", f"{cagr_values['net_income_cagr']:.2%}" if cagr_values['net_income_cagr'] is not None else "N/A"],
        ["Total Assets CAGR", f"{cagr_values['assets_cagr']:.2%}" if cagr_values['assets_cagr'] is not None else "N/A"],
        ["Total Liabilities CAGR", f"{cagr_values['liabilities_cagr']:.2%}" if cagr_values['liabilities_cagr'] is not None else "N/A"],
        ["Operating Cash Flow CAGR", f"{cagr_values['cashflow_cagr']:.2%}" if cagr_values['cashflow_cagr'] is not None else "N/A"],
        ["Factor 3 Score (Time Series Analysis)", f"{factor_scores['factor3_score']}"],
    ]
    cagr_table = Table(cagr_data, hAlign='LEFT')
    cagr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    elements.append(cagr_table)
    elements.append(Spacer(1, 12))

    # Sentiment Analysis
    elements.append(Paragraph("Sentiment Analysis", styles['Heading1']))

    for sentiment in sentiment_results:
        elements.append(Paragraph(sentiment['title'], styles['Heading2']))
        sentiment_text = f"""
        - Sentiment Score: {sentiment['score']:.2f} <br/>
        - Explanation: {sentiment['explanation']} <br/>
        - Factor Score: {sentiment['factor_score']}
        """
        elements.append(Paragraph(sentiment_text, styles['Normal']))
        elements.append(Spacer(1, 12))

    # Data Visualizations
    elements.append(Paragraph("Data Visualizations", styles['Heading1']))
    for plot_title, plot_image in plots.items():
        elements.append(Paragraph(plot_title, styles['Heading2']))
        img = Image(plot_image, width=500, height=200)
        elements.append(img)
        elements.append(Spacer(1, 12))

    # Final Recommendation
    elements.append(Paragraph("Final Recommendation", styles['Heading1']))
    recommendation_text = f"""
    The weighted total score based on the analysis is: {weighted_total_score}.<br/>
    The final recommendation is: <b>{recommendation}</b>.
    """
    elements.append(Paragraph(recommendation_text, styles['Normal']))

    doc.build(elements)
    logger.debug('PDF report generation completed.')

    pdf_output.seek(0)
    return pdf_output