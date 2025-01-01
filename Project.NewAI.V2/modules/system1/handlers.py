import os
import io
import time
import asyncio
import logging
import openai
from openai.error import APIError, RateLimitError
import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
from flask import request, send_file, render_template, jsonify, Blueprint, redirect, url_for
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from asgiref.wsgi import WsgiToAsgi  # For ASGI compatibility
import config

import boto3
import uuid
import tempfile
import re
import nltk
from datetime import datetime
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import tiktoken
import openai
from pinecone import Pinecone, ServerlessSpec
from logging.handlers import RotatingFileHandler
from modules.system1.alpha_vantage_service import get_annual_price_change

# NEW IMPORTS FOR SMART LB & MODEL SELECTOR
from model_selector import choose_model_for_task
from smart_load_balancer import call_openai_smart_async

# IMPORT ANALYSIS STORAGE FUNCTIONS
from analysis_storage import store_results_for_user, get_results_for_user

# Blueprint
system1_bp = Blueprint('system1_bp', __name__, template_folder='templates')

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = config.UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    'csv': {'csv'},
    'pdf': {'pdf'},
    'image': {'png', 'jpg', 'jpeg', 'gif'}
}

# AWS
S3_BUCKET_NAME = config.S3_BUCKET_NAME
s3_client = boto3.client('s3', region_name=config.AWS_REGION)

# OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
openai.api_key = openai_api_key

# PINECONE (if needed for something else)
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENVIRONMENT = config.PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME

# Replaced the old pinecone.init(...) with the Pinecone 5.x class approach
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # or whatever dimension your embeddings are
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# NLTK config
NLTK_DATA_PATH = os.path.join(os.path.expanduser('~'), 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)
try:
    nltk.data.find('tokenizers/punkt')
    logger.debug("NLTK 'punkt' tokenizer found")
except LookupError:
    logger.debug("NLTK 'punkt' tokenizer not found, downloading...")
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    logger.debug("NLTK 'punkt' tokenizer downloaded")

# Flask app (if needed)
from flask import Flask
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'YOUR_FLASK_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

###############################################################################
# HELPERS
###############################################################################

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning None if denominator is zero."""
    try:
        result = numerator / denominator if denominator != 0 else None
        logger.debug(f'Safe divide {numerator} / {denominator} = {result}')
        return result
    except Exception as e:
        logger.error(f'Error in safe_divide: {str(e)}')
        return None

def dcf_analysis(projected_free_cash_flows, wacc, terminal_value, projection_years):
    """Discounted Cash Flow analysis"""
    logger.debug('Starting DCF analysis.')
    discounted_free_cash_flows = [
        fcf / (1 + wacc) ** i for i, fcf in enumerate(projected_free_cash_flows, 1)
    ]
    discounted_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
    dcf_value = sum(discounted_free_cash_flows) + discounted_terminal_value
    logger.debug('Completed DCF analysis.')
    return dcf_value

def calculate_ratios(financials, benchmarks):
    """Calculate key financial ratios and compare them to benchmarks."""
    logger.debug('Starting ratio analysis.')
    current_ratio = safe_divide(financials['current_assets'], financials['current_liabilities'])
    debt_to_equity = safe_divide(financials['total_debt'], financials['shareholders_equity'])
    pe_ratio = safe_divide(financials['market_price'], financials['eps'])
    pb_ratio = safe_divide(financials['market_price'], financials['book_value_per_share'])

    scores = {}
    if debt_to_equity is not None:
        scores['debt_to_equity'] = 1 if debt_to_equity < benchmarks['debt_to_equity'] else -1
    else:
        scores['debt_to_equity'] = 0

    if current_ratio is not None:
        scores['current_ratio'] = 1 if current_ratio > benchmarks['current_ratio'] else -1
    else:
        scores['current_ratio'] = 0

    if pe_ratio is not None:
        scores['pe_ratio'] = 1 if pe_ratio < benchmarks['pe_ratio'] else -1
    else:
        scores['pe_ratio'] = 0

    if pb_ratio is not None:
        scores['pb_ratio'] = 1 if pb_ratio < benchmarks['pb_ratio'] else -1
    else:
        scores['pb_ratio'] = 0

    total_score = sum(scores.values())
    logger.debug(f'Ratio analysis scores: {scores}, Total Score: {total_score}')

    if total_score >= 2:
        normalized_factor2_score = 1
    elif total_score <= -2:
        normalized_factor2_score = -1
    else:
        normalized_factor2_score = 0

    logger.debug(f'Normalized Factor 2 Score: {normalized_factor2_score}')

    return {
        'Current Ratio': current_ratio,
        'Debt-to-Equity Ratio': debt_to_equity,
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'Scores': scores,
        'Total Score': total_score,
        'Normalized Factor 2 Score': normalized_factor2_score
    }

def calculate_cagr(beginning_value, ending_value, periods):
    """Calculate the Compound Annual Growth Rate."""
    try:
        if beginning_value <= 0 or periods <= 0:
            logger.warning('Invalid beginning value or periods for CAGR calculation.')
            return None
        cagr = (ending_value / beginning_value) ** (1 / periods) - 1
        logger.debug(f'Calculated CAGR: {cagr}')
        return cagr
    except Exception as e:
        logger.error(f'Error calculating CAGR: {str(e)}')
        return None

###############################################################################
# PLOTTING FUNCTIONS
###############################################################################

def generate_plot(x, y, title, x_label, y_label):
    logger.debug(f'Generating plot: {title}')
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    img_data = io.BytesIO()
    plt.savefig(img_data, format='PNG')
    plt.close()
    img_data.seek(0)
    logger.debug(f'Plot generated: {title}')
    return img_data

def generate_benchmark_comparison_plot(benchmark_comparison):
    labels = benchmark_comparison['Ratios']
    company_values = benchmark_comparison['Company']
    industry_values = benchmark_comparison['Industry']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_company = ax.bar(x - width/2, company_values, width, label='Company')
    bars_industry = ax.bar(x + width/2, industry_values, width, label='Industry')
    ax.set_ylabel('Ratio Values')
    ax.set_title('Company vs. Industry Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    autolabel(bars_company)
    autolabel(bars_industry)

    fig.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='PNG', bbox_inches='tight')
    plt.close(fig)
    img_data.seek(0)
    return img_data

###############################################################################
# PDF & AI SUMMARIZATION / SENTIMENT LOGIC
###############################################################################

def extract_text_from_pdf(file_path):
    logger.debug(f'Extracting text from PDF: {file_path}')
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num, page in enumerate(reader.pages, 1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                else:
                    logger.warning(f'No text found on page {page_num} of {file_path}.')
            if not text.strip():
                logger.warning(f'No extractable text found in {file_path}.')
                return None
            logger.debug(f'Text extracted from PDF: {file_path}')
            return text
    except Exception as e:
        logger.error(f'Error reading PDF file {file_path}: {str(e)}')
        return None

semaphore = asyncio.Semaphore(5)

async def call_openai_summarization(text):
    retry_delay = 5
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model_for_summarization = choose_model_for_task("short_summarization")
            response = await call_openai_smart_async(
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                model=model_for_summarization,
                temperature=0.5,
                max_tokens=500
            )
            summary = response["choices"][0]["message"]["content"].strip()
            return summary

        except RateLimitError:
            logger.warning(f'Rate limit error, retrying in {retry_delay} seconds...')
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
        except APIError as e:
            logger.error(f'OpenAI API error: {str(e)}')
            return None
        except Exception as e:
            logger.error(f'Unexpected error during OpenAI API call: {str(e)}')
            return None
    logger.error('Failed to summarize text after multiple attempts.')
    return None

async def call_openai_analyze_sentiment(text, context):
    retry_delay = 5
    max_retries = 5
    prompt = f"""
As an expert financial analyst, analyze the following {context}.
Provide a sentiment score between -1 (very negative) and 1 (very positive).
Also, briefly explain the main factors contributing to this sentiment.

Text:
{text}

Response Format:
Sentiment Score: [score]
Explanation: [brief explanation]
"""
    for attempt in range(max_retries):
        try:
            logger.debug(f'Attempting sentiment analysis, attempt {attempt+1}.')

            model_for_sentiment = choose_model_for_task("short_summarization")
            response = await call_openai_smart_async(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in sentiment analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=model_for_sentiment,
                temperature=0.5,
                max_tokens=500
            )

            content = response["choices"][0]["message"]["content"].strip()
            lines = content.split('\n')
            sentiment_score = None
            explanation = ""
            for line in lines:
                if "Sentiment Score:" in line:
                    try:
                        sentiment_score = float(line.split("Sentiment Score:")[1].strip())
                    except ValueError:
                        sentiment_score = None
                elif "Explanation:" in line:
                    explanation = line.split("Explanation:", 1)[1].strip()
            return sentiment_score, explanation

        except RateLimitError:
            logger.warning(f'Rate limit exceeded during sentiment analysis. Retrying in {retry_delay} seconds...')
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
        except APIError as e:
            logger.error(f'OpenAI API error during sentiment analysis: {str(e)}')
            return None, ""
        except Exception as e:
            logger.error(f'Error performing sentiment analysis: {str(e)}')
            return None, ""
    logger.error('Failed to perform sentiment analysis after multiple attempts.')
    return None, ""

def run_async_function(coroutine):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coroutine)
        loop.close()
        return result

def parse_headings_in_10k(text):
    pattern = r'(ITEM\s+[0-9A-Z]+(?:\.[0-9A-Z]+)*)'
    splits = re.split(pattern, text, flags=re.IGNORECASE)

    sections = []
    current_heading = "INTRODUCTION"
    current_text_parts = []

    for i, segment in enumerate(splits):
        segment = segment.strip()
        if i == 0:
            if segment:
                current_text_parts.append(segment)
        else:
            if re.match(pattern, segment, flags=re.IGNORECASE):
                joined_text = "\n".join(current_text_parts).strip()
                if joined_text:
                    sections.append((current_heading, joined_text))
                current_text_parts = []
                current_heading = segment
            else:
                current_text_parts.append(segment)
    leftover = "\n".join(current_text_parts).strip()
    if leftover:
        sections.append((current_heading, leftover))

    return sections

def parse_10k_by_headings(ten_k_text):
    all_sections = parse_headings_in_10k(ten_k_text)
    result = {"item1": "", "item1A": "", "item7": ""}

    for heading, content in all_sections:
        heading_upper = heading.upper()
        if "ITEM 1A" in heading_upper or "RISK FACTORS" in heading_upper:
            result["item1A"] += f"\n{content}"
        elif "ITEM 1" in heading_upper and "1A" not in heading_upper:
            result["item1"] += f"\n{content}"
        elif "ITEM 7" in heading_upper or "MANAGEMENT'S DISCUSSION" in heading_upper:
            result["item7"] += f"\n{content}"

    return result

async def summarize_text_async(text):
    logger.debug('Starting text summarization (aggressive approach).')
    max_tokens_per_chunk = 3800
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    token_count = len(tokens)

    if token_count <= max_tokens_per_chunk:
        summary = await call_openai_summarization(text)
        return summary

    chunks = []
    for i in range(0, token_count, max_tokens_per_chunk):
        chunk_tokens = tokens[i:i + max_tokens_per_chunk]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    tasks = [call_openai_summarization(chunk) for chunk in chunks]
    summaries = await asyncio.gather(*tasks)
    combined_summary = ' '.join(filter(None, summaries))
    logger.debug('Text summarization completed.')
    return combined_summary

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
        ["Debt-to-Equity Ratio", f"{ratios['Debt-to-Equity Ratio']:.2f}"],
        ["Current Ratio", f"{ratios['Current Ratio']:.2f}"],
        ["P/E Ratio", f"{ratios['P/E Ratio']:.2f}"],
        ["P/B Ratio", f"{ratios['P/B Ratio']:.2f}"],
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

###############################################################################
# CSV PROCESSING
###############################################################################

def process_financial_csv(file_path, csv_name):
    logger.debug(f'Processing CSV file: {file_path}')
    try:
        df = pd.read_csv(file_path, index_col=0, thousands=',', quotechar='"')
        df.columns = df.columns.str.strip()
        df.index = df.index.str.strip()
        if 'ttm' in df.columns:
            df.drop(columns=['ttm'], inplace=True)
        df = df.transpose()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df = df[df['Date'].notnull()]
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)
        logger.debug(f"Processed {csv_name} CSV columns: {df.columns.tolist()}")
        logger.debug(f"{csv_name} DataFrame after processing:\n{df.head()}")
        logger.debug(f'Completed processing CSV file: {file_path}')
        return df
    except Exception as e:
        logger.error(f"Error processing {csv_name} CSV: {str(e)}")
        return None

def standardize_columns(df, column_mappings, csv_name):
    logger.debug(f'Standardizing columns for {csv_name}.')
    df_columns = df.columns.tolist()
    new_columns = {}
    for standard_name, possible_names in column_mappings.items():
        for name in possible_names:
            if name in df_columns:
                new_columns[name] = standard_name
                break
    df.rename(columns=new_columns, inplace=True)
    logger.debug(f'After standardization, columns in {csv_name}: {df.columns.tolist()}')
    return df

###############################################################################
# MAIN ROUTE (ANALYZE)
###############################################################################

@system1_bp.route('/analyze', methods=['POST'])
def analyze_financials():
    """
    Main entry point for the aggressive approach:
    - Extract only ITEM 1, 1A, 7 from the 10-K
    - Summarize each item once
    - Perform sentiment on those summaries
    - Do CSV-based DCF analysis, ratio analysis, time-series
    - Generate final PDF
    """
    try:
        logger.info('Starting analyze_financials function.')
        file_paths = {}
        expected_files = {
            'income_statement': 'csv',
            'balance_sheet': 'csv',
            'cash_flow': 'csv',
            'earnings_call': 'pdf',
            'industry_report': 'pdf',
            'economic_report': 'pdf',
            'ten_k_report': 'pdf',
            'company_logo': 'image'
        }
        for field_name, file_type in expected_files.items():
            uploaded_file = request.files.get(field_name)
            if uploaded_file and allowed_file(uploaded_file.filename, ALLOWED_EXTENSIONS[file_type]):
                filename = secure_filename(uploaded_file.filename)
                if field_name == 'company_logo':
                    file_path = os.path.join(UPLOAD_FOLDER, f"logo_{filename}")
                else:
                    file_path = os.path.join(UPLOAD_FOLDER, f"{field_name}_{filename}")
                uploaded_file.save(file_path)
                file_paths[field_name] = file_path
                logger.info(f"Received file: {field_name} - {filename}")
            else:
                if field_name == 'company_logo':
                    file_paths[field_name] = None
                    logger.warning('No company logo uploaded or invalid file type.')
                else:
                    error_message = f'Invalid or missing file for {field_name}. Please upload a valid {file_type.upper()} file.'
                    logger.warning(error_message)
                    return jsonify({'error': error_message}), 400

        # Extract text from the 10-K
        logger.debug('Extracting text from the 10-K report.')
        ten_k_text = extract_text_from_pdf(file_paths['ten_k_report'])
        if not ten_k_text:
            error_message = 'Error extracting text from the 10-K report. Ensure the PDF contains extractable text.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400
        else:
            logger.info('Extracted text from the 10-K report successfully.')

        # Parse only ITEM 1, 1A, 7
        items_dict = parse_10k_by_headings(ten_k_text)

        # Summaries
        summaries = {}
        for key in ["item1", "item1A", "item7"]:
            item_text = items_dict[key]
            if item_text.strip():
                item_summary = run_async_function(summarize_text_async(item_text))
                summaries[key] = item_summary if item_summary else ""
            else:
                summaries[key] = ""
                logger.debug(f"No text found for {key} in the 10-K.")

        company_summary = summaries["item1"]
        risks_summary = summaries["item1A"]
        industry_summary = summaries["item7"]

        # Summarize + sentiment for other PDFs
        def read_and_summarize(pdf_path):
            if not pdf_path:
                return "", (0, "No file provided")
            text_ = extract_text_from_pdf(pdf_path)
            if not text_:
                return "", (0, "No text extracted")
            summ_ = run_async_function(summarize_text_async(text_))
            if not summ_:
                summ_ = ""
            score_, expl_ = run_async_function(call_openai_analyze_sentiment(summ_, "pdf summary"))
            return summ_, (score_ if score_ else 0, expl_)

        ecall_summary, ecall_sentiment = read_and_summarize(file_paths['earnings_call'])
        ireport_summary, ireport_sentiment = read_and_summarize(file_paths['industry_report'])
        ereport_summary, ereport_sentiment = read_and_summarize(file_paths['economic_report'])

        earnings_call_score, earnings_call_explanation = ecall_sentiment
        industry_report_score, industry_report_explanation = ireport_sentiment
        economic_report_score, economic_report_explanation = ereport_sentiment

        # Get the industry from user form
        industry_name = request.form.get('industry_name', 'Software')
        company_name = request.form.get('company_name', 'N/A')
        wacc = float(request.form['wacc']) / 100
        tax_rate = float(request.form['tax_rate']) / 100
        growth_rate = float(request.form['growth_rate']) / 100
        stock_price = float(request.form['stock_price'])

        debt_equity_benchmark = float(request.form['debt_equity_benchmark'])
        current_ratio_benchmark = float(request.form['current_ratio_benchmark'])
        pe_benchmark = float(request.form['pe_benchmark'])
        pb_benchmark = float(request.form['pb_benchmark'])

        benchmarks = {
            'debt_to_equity': debt_equity_benchmark,
            'current_ratio': current_ratio_benchmark,
            'pe_ratio': pe_benchmark,
            'pb_ratio': pb_benchmark
        }

        ########################################################################
        # STEP B: Retrieve Industry Standard Margin via GPT (or static fallback)
        ########################################################################
        def parse_float_from_gpt(gpt_text):
            """Extract a floating percentage from GPT text (very naive approach)."""
            import re
            # e.g. GPT might respond: 'Typical margin is around 18%.'
            match = re.search(r'([\d.]+)\s?%', gpt_text)
            if match:
                return float(match.group(1)) / 100.0
            return 0.15  # fallback if not found

        gpt_prompt = f"What is the typical profit margin in the {industry_name} industry, as a percentage?"
        try:
            from smart_load_balancer import call_openai_smart
            gpt_response = call_openai_smart(
                messages=[{"role": "user", "content": gpt_prompt}],
                model="gpt-3.5-turbo",
                temperature=0.0,
                max_tokens=100
            )
            gpt_text = gpt_response['choices'][0]['message']['content']
            industry_profit_margin = parse_float_from_gpt(gpt_text)
        except Exception as e:
            logger.warning(f"Could not retrieve industry margin from GPT, using fallback: {e}")
            industry_profit_margin = 0.15
        logger.debug(f"Industry margin estimated at: {industry_profit_margin}")

        # Process CSV
        income_df = process_financial_csv(file_paths['income_statement'], 'income_statement')
        balance_df = process_financial_csv(file_paths['balance_sheet'], 'balance_sheet')
        cashflow_df = process_financial_csv(file_paths['cash_flow'], 'cash_flow')
        if income_df is None or balance_df is None or cashflow_df is None:
            error_message = 'Error processing CSV files.'
            logger.error(error_message)
            return jsonify({'error': error_message}), 400

        income_columns = {
            'Revenue': ['TotalRevenue', 'Revenue', 'Total Revenue', 'Sales'],
            'Net Income': ['NetIncome', 'Net Income', 'Net Profit', 'Profit After Tax'],
        }
        balance_columns = {
            'Total Assets': ['TotalAssets', 'Total Assets'],
            'Total Liabilities': ['TotalLiabilities', 'Total Liabilities', 'TotalLiabilitiesNetMinorityInterest'],
            'Shareholders Equity': ['TotalEquity', 'Shareholders Equity', 'Total Equity', 'StockholdersEquity'],
            'Current Assets': ['CurrentAssets', 'Current Assets'],
            'Current Liabilities': ['CurrentLiabilities', 'Current Liabilities'],
            'CurrentDebt': ['CurrentDebt', 'ShortTermDebt', 'Short-Term Debt', 'Current Debt'],
            'Long-Term Debt': ['LongTermDebt', 'Long-Term Debt', 'Non-Current Debt', 'Long-Term Debt'],
            'Total Shares Outstanding': ['SharesIssued', 'ShareIssued', 'Total Shares Outstanding', 'Total Shares', 'OrdinarySharesNumber'],
            'Inventory': ['Inventory', 'Inventories'],
        }
        cashflow_columns = {
            'Operating Cash Flow': ['OperatingCashFlow', 'Operating Cash Flow', 'Cash from Operations'],
            'Capital Expenditures': ['CapitalExpenditures', 'Capital Expenditures', 'CapEx', 'CapitalExpenditure', 'Capital Expenditure'],
        }

        income_df = standardize_columns(income_df, income_columns, 'income_statement')
        balance_df = standardize_columns(balance_df, balance_columns, 'balance_sheet')
        cashflow_df = standardize_columns(cashflow_df, cashflow_columns, 'cash_flow')

        for df_obj, name in [
            (income_df, 'income_statement'),
            (balance_df, 'balance_sheet'),
            (cashflow_df, 'cash_flow')
        ]:
            if 'Date' not in df_obj.columns:
                error_message = f"'Date' column missing in {name} CSV."
                logger.error(error_message)
                return jsonify({'error': error_message}), 400

        income_df.sort_values('Date', inplace=True)
        balance_df.sort_values('Date', inplace=True)
        cashflow_df.sort_values('Date', inplace=True)

        income_df.reset_index(drop=True, inplace=True)
        balance_df.reset_index(drop=True, inplace=True)
        cashflow_df.reset_index(drop=True, inplace=True)

        numeric_columns = [
            'Revenue', 'Net Income', 'Total Assets', 'Total Liabilities',
            'Shareholders Equity', 'Current Assets', 'Current Liabilities',
            'CurrentDebt', 'Long-Term Debt', 'Total Shares Outstanding',
            'Operating Cash Flow', 'Capital Expenditures', 'Inventory'
        ]
        for col in numeric_columns:
            if col in income_df.columns:
                income_df[col] = pd.to_numeric(income_df[col], errors='coerce')
            if col in balance_df.columns:
                balance_df[col] = pd.to_numeric(balance_df[col], errors='coerce')
            if col in cashflow_df.columns:
                cashflow_df[col] = pd.to_numeric(cashflow_df[col], errors='coerce')

        income_df.fillna(0, inplace=True)
        balance_df.fillna(0, inplace=True)
        cashflow_df.fillna(0, inplace=True)
        logger.info('Processed CSV files successfully.')

        ########################################################################
        # STEP A: Compute Annual Profit Margins for the Last 4 Years
        ########################################################################
        all_years = sorted(income_df['Date'].dt.year.unique())
        if len(all_years) > 4:
            last_4_years = all_years[-4:]
        else:
            last_4_years = all_years

        annual_profit_margins = {}
        for yr in last_4_years:
            rows_for_year = income_df[income_df['Date'].dt.year == yr]
            total_revenue_yr = rows_for_year['Revenue'].sum()
            total_net_income_yr = rows_for_year['Net Income'].sum()
            if total_revenue_yr > 0:
                pm = total_net_income_yr / total_revenue_yr
            else:
                pm = 0
            annual_profit_margins[str(yr)] = pm
        logger.debug(f"Computed annual profit margins: {annual_profit_margins}")

        latest_date = balance_df['Date'].max()
        if 'Long-Term Debt' in balance_df.columns:
            long_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'Long-Term Debt'].values[0]
        else:
            long_term_debt = 0
            logger.warning("Long-Term Debt not found in balance sheet columns.")

        if 'CurrentDebt' in balance_df.columns:
            short_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'CurrentDebt'].values[0]
        else:
            short_term_debt = 0
            logger.warning("Current Debt not found in balance sheet columns.")

        total_debt = long_term_debt + short_term_debt
        shareholders_equity = balance_df.loc[balance_df['Date'] == latest_date, 'Shareholders Equity'].values[0]
        current_assets = balance_df.loc[balance_df['Date'] == latest_date, 'Current Assets'].values[0]
        current_liabilities = balance_df.loc[balance_df['Date'] == latest_date, 'Current Liabilities'].values[0]
        total_shares_outstanding = balance_df.loc[balance_df['Date'] == latest_date, 'Total Shares Outstanding'].values[0] if 'Total Shares Outstanding' in balance_df.columns else 1
        inventory = balance_df.loc[balance_df['Date'] == latest_date, 'Inventory'].values[0] if 'Inventory' in balance_df.columns else 0
        book_value_per_share = safe_divide(shareholders_equity, total_shares_outstanding)

        net_income = income_df['Net Income'].iloc[-1]
        revenue = income_df['Revenue'].iloc[-1]
        eps = safe_divide(net_income, total_shares_outstanding)

        if 'Operating Cash Flow' not in cashflow_df.columns or 'Capital Expenditures' not in cashflow_df.columns:
            error_message = 'Missing Operating Cash Flow or Capital Expenditures column in the cash flow statement.'
            logger.error(error_message)
            return jsonify({'error': error_message}), 400

        operating_cash_flow = cashflow_df['Operating Cash Flow']
        capital_expenditures = cashflow_df['Capital Expenditures']
        free_cash_flows = operating_cash_flow - capital_expenditures
        free_cash_flows = free_cash_flows.reset_index(drop=True)

        if len(free_cash_flows) < 2:
            error_message = 'Not enough data to project future free cash flows.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        historical_growth_rates = free_cash_flows.pct_change().dropna()
        average_growth_rate = historical_growth_rates.mean()
        logger.debug(f'Average historical growth rate: {average_growth_rate}')
        projected_growth_rate = min(average_growth_rate, growth_rate)
        logger.debug(f'Projected growth rate used: {projected_growth_rate}')

        projection_years = 5
        last_free_cash_flow = free_cash_flows.iloc[-1]
        projected_free_cash_flows = [
            last_free_cash_flow * (1 + projected_growth_rate) ** i
            for i in range(1, projection_years + 1)
        ]
        if wacc <= growth_rate:
            error_message = 'WACC must be greater than the growth rate for DCF calculation.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        terminal_value = projected_free_cash_flows[-1] * (1 + growth_rate) / (wacc - growth_rate)
        dcf_value = dcf_analysis(projected_free_cash_flows, wacc, terminal_value, projection_years)
        intrinsic_value_per_share = safe_divide(dcf_value, total_shares_outstanding)

        if intrinsic_value_per_share is None:
            factor1_score = 0
            logger.warning('Intrinsic value per share could not be calculated.')
        else:
            upper_bound = stock_price * 1.10
            lower_bound = stock_price * 0.90
            if intrinsic_value_per_share > upper_bound:
                factor1_score = 1
            elif intrinsic_value_per_share < lower_bound:
                factor1_score = -1
            else:
                factor1_score = 0

        financials = {
            'company_name': company_name,
            'total_debt': total_debt,
            'shareholders_equity': shareholders_equity,
            'current_assets': current_assets,
            'current_liabilities': current_liabilities,
            'inventory': inventory,
            'market_price': stock_price,
            'eps': eps,
            'book_value_per_share': book_value_per_share,
            'net_income': net_income,
            'revenue': revenue
        }
        logger.info('Extracted financial data successfully.')

        # Calculate your existing ratios
        ratios = calculate_ratios(financials, benchmarks)
        factor2_score = ratios['Normalized Factor 2 Score']

        ########################################################################
        # STEP 3: Fetch Industry Benchmarks from GPT (Optional), Compute More Ratios, Time-Series
        ########################################################################
        def parse_industry_benchmarks_from_gpt(gpt_text):
            """
            Attempts to parse JSON from GPT text with keys like:
              "profit_margin", "current_ratio", "debt_equity", "roa"
            Returns a dict or a fallback if parsing fails.
            """
            import json
            import re
            pattern = r"\{.*\}"
            match = re.search(pattern, gpt_text.strip())
            if not match:
                # fallback
                return {
                    "profit_margin": 0.15,
                    "current_ratio": 1.2,
                    "debt_equity": 0.8,
                    "roa": 0.07
                }
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return {
                    "profit_margin": float(data.get("profit_margin", 0.15)),
                    "current_ratio": float(data.get("current_ratio", 1.2)),
                    "debt_equity": float(data.get("debt_equity", 0.8)),
                    "roa": float(data.get("roa", 0.07))
                }
            except Exception:
                return {
                    "profit_margin": 0.15,
                    "current_ratio": 1.2,
                    "debt_equity": 0.8,
                    "roa": 0.07
                }

        benchmark_prompt = f"""
Given the '{industry_name}' industry, what are typical benchmarks for:
 - profit_margin (as a fraction, e.g. 0.15 for 15%)
 - current_ratio
 - debt_equity
 - roa

Please provide them in valid JSON only, like:
{{"profit_margin": 0.15, "current_ratio": 1.2, "debt_equity": 0.8, "roa": 0.07}}
"""
        try:
            from smart_load_balancer import call_openai_smart
            bench_gpt_response = call_openai_smart(
                messages=[{"role": "user", "content": benchmark_prompt}],
                model="gpt-3.5-turbo",
                temperature=0.0,
                max_tokens=200
            )
            bench_gpt_text = bench_gpt_response['choices'][0]['message']['content']
            industry_benchmarks_dict = parse_industry_benchmarks_from_gpt(bench_gpt_text)
        except Exception as e:
            logger.warning(f"Could not retrieve industry benchmarks from GPT, using fallback: {e}")
            industry_benchmarks_dict = {
                "profit_margin": 0.15,
                "current_ratio": 1.2,
                "debt_equity": 0.8,
                "roa": 0.07
            }
        logger.debug(f"Industry Benchmarks (detailed): {industry_benchmarks_dict}")

        # Additional ratio calculations
        profit_margin_calc = safe_divide(net_income, revenue)  # fraction
        current_ratio_calc = safe_divide(current_assets, current_liabilities)
        debt_equity_calc = safe_divide(total_debt, shareholders_equity)
        total_assets_latest = balance_df.loc[balance_df['Date'] == latest_date, 'Total Assets'].values[0] if 'Total Assets' in balance_df.columns else 0
        roa_calc = safe_divide(net_income, total_assets_latest)

        # Store them inside "ratios" or a new dict
        ratios["profit_margin"] = profit_margin_calc
        ratios["current_ratio_calc"] = current_ratio_calc
        ratios["debt_equity_calc"] = debt_equity_calc
        ratios["roa_calc"] = roa_calc

        # Optional: store the new industry benchmarks
        final_industry_bench = {
            "profit_margin": industry_benchmarks_dict["profit_margin"],
            "current_ratio": industry_benchmarks_dict["current_ratio"],
            "debt_equity": industry_benchmarks_dict["debt_equity"],
            "roa": industry_benchmarks_dict["roa"]
        }

        # For optional expanded time-series: last 4 quarters
        latest_quarters_df = income_df.sort_values('Date', ascending=False).head(4).copy()
        latest_quarters_df.sort_values('Date', ascending=True, inplace=True)
        quarterly_data = []
        for i, row in latest_quarters_df.iterrows():
            quarter_label = row['Date'].strftime('%Y-Q') + str((row['Date'].month-1)//3 + 1)
            rev = row['Revenue']
            ni = row['Net Income']
            pm_ = safe_divide(ni, rev)
            quarterly_data.append({
                "quarter": quarter_label,
                "revenue": rev,
                "net_income": ni,
                "profit_margin": pm_
            })

        # Time to do the factor3_time_series or keep it in final
        n_periods = len(income_df) - 1
        if n_periods < 1:
            error_message = 'Not enough data for time series analysis.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        revenue_cagr = calculate_cagr(income_df['Revenue'].iloc[0], income_df['Revenue'].iloc[-1], n_periods)
        net_income_cagr = calculate_cagr(income_df['Net Income'].iloc[0], income_df['Net Income'].iloc[-1], n_periods)
        assets_cagr = calculate_cagr(balance_df['Total Assets'].iloc[0], balance_df['Total Assets'].iloc[-1], n_periods)
        liabilities_cagr = calculate_cagr(balance_df['Total Liabilities'].iloc[0], balance_df['Total Liabilities'].iloc[-1], n_periods)
        cashflow_cagr = calculate_cagr(cashflow_df['Operating Cash Flow'].iloc[0], cashflow_df['Operating Cash Flow'].iloc[-1], n_periods)

        factor3_scores = []
        if revenue_cagr is not None:
            factor3_scores.append(1 if revenue_cagr > 0 else -1)
        if net_income_cagr is not None:
            factor3_scores.append(1 if net_income_cagr > 0 else -1)
        if assets_cagr is not None:
            factor3_scores.append(1 if assets_cagr > 0 else -1)
        if liabilities_cagr is not None:
            factor3_scores.append(-1 if liabilities_cagr > 0 else 1)
        if cashflow_cagr is not None:
            factor3_scores.append(1 if cashflow_cagr > 0 else -1)

        factor3_score = sum(factor3_scores)
        if factor3_score > 0:
            factor3_score = 1
        elif factor3_score < 0:
            factor3_score = -1
        else:
            factor3_score = 0

        # Generate plots
        plots = {}
        benchmark_comparison = {
            'Ratios': ['Debt-to-Equity Ratio', 'Current Ratio', 'P/E Ratio', 'P/B Ratio'],
            'Company': [
                ratios.get('Debt-to-Equity Ratio', 0) if ratios.get('Debt-to-Equity Ratio') is not None else 0,
                ratios.get('Current Ratio', 0) if ratios.get('Current Ratio') is not None else 0,
                ratios.get('P/E Ratio', 0) if ratios.get('P/E Ratio') is not None else 0,
                ratios.get('P/B Ratio', 0) if ratios.get('P/B Ratio') is not None else 0,
            ],
            'Industry': [
                benchmarks['debt_to_equity'],
                benchmarks['current_ratio'],
                benchmarks['pe_ratio'],
                benchmarks['pb_ratio'],
            ]
        }
        benchmark_plot = generate_benchmark_comparison_plot(benchmark_comparison)
        plots['Company vs. Industry Benchmarks'] = benchmark_plot

        dates = income_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        plots['Revenue Over Time'] = generate_plot(
            dates,
            income_df['Revenue'].tolist(),
            'Revenue Over Time',
            'Date',
            'Revenue'
        )
        plots['Net Income Over Time'] = generate_plot(
            dates,
            income_df['Net Income'].tolist(),
            'Net Income Over Time',
            'Date',
            'Net Income'
        )

        cashflow_dates = cashflow_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        plots['Operating Cash Flow Over Time'] = generate_plot(
            cashflow_dates,
            cashflow_df['Operating Cash Flow'].tolist(),
            'Operating Cash Flow Over Time',
            'Date',
            'Operating Cash Flow'
        )

        balance_dates = balance_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        plt.figure(figsize=(8, 4))
        plt.plot(balance_dates, balance_df['Total Assets'].tolist(), marker='o', label='Total Assets')
        plt.plot(balance_dates, balance_df['Total Liabilities'].tolist(), marker='o', label='Total Liabilities')
        plt.title('Total Assets and Liabilities Over Time')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='PNG')
        plt.close()
        img_data.seek(0)
        plots['Total Assets and Liabilities Over Time'] = img_data

        # Factor Scores
        def map_sentiment_to_score(sentiment_score):
            if sentiment_score is None:
                return 0
            elif sentiment_score > 0.5:
                return 1
            elif sentiment_score < -0.5:
                return -1
            else:
                return 0

        factor4_score = map_sentiment_to_score(earnings_call_score)
        factor5_score = map_sentiment_to_score(industry_report_score)
        factor6_score = map_sentiment_to_score(economic_report_score)

        weights = {
            'factor1': 1,
            'factor2': 1,
            'factor3': 1,
            'factor4': 1,
            'factor5': 1,
            'factor6': 1
        }

        weighted_total_score = (
            factor1_score * weights['factor1'] +
            factor2_score * weights['factor2'] +
            factor3_score * weights['factor3'] +
            factor4_score * weights['factor4'] +
            factor5_score * weights['factor5'] +
            factor6_score * weights['factor6']
        )
        if weighted_total_score >= 3:
            recommendation = "Buy"
        elif weighted_total_score <= -3:
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        sentiment_results = [
            {
                'title': 'Earnings Call Sentiment',
                'score': earnings_call_score,
                'explanation': earnings_call_explanation,
                'factor_score': factor4_score
            },
            {
                'title': 'Industry Report Sentiment',
                'score': industry_report_score,
                'explanation': industry_report_explanation,
                'factor_score': factor5_score
            },
            {
                'title': 'Economic Report Sentiment',
                'score': economic_report_score,
                'explanation': economic_report_explanation,
                'factor_score': factor6_score
            },
        ]

        factor_scores = {
            'factor1_score': factor1_score,
            'factor2_score': factor2_score,
            'factor3_score': factor3_score,
            'factor4_score': factor4_score,
            'factor5_score': factor5_score,
            'factor6_score': factor6_score,
        }

        # Build final_analysis dict to store
        composite_score = 0
        score_count = 0
        for s in [earnings_call_score, industry_report_score, economic_report_score]:
            if s is not None:
                composite_score += s
                score_count += 1
        if score_count > 0:
            composite_score /= score_count

        final_analysis = {
            "company_report": {
                "stock_price": stock_price,
                "executive_summary": company_summary,
                "company_summary": company_summary,
                "industry_summary": industry_summary,
                "risk_considerations": risks_summary
            },
            "financial_analysis": {
                "dcf_intrinsic_value": intrinsic_value_per_share or 0,
                "ratios": ratios,
                "time_series_analysis": {
                    "revenue_cagr": revenue_cagr,
                    "net_income_cagr": net_income_cagr,
                    "assets_cagr": assets_cagr,
                    "liabilities_cagr": liabilities_cagr,
                    "cashflow_cagr": cashflow_cagr
                }
            },
            "sentiment": {
                "composite_score": composite_score,
                "earnings_call_sentiment": {
                    "score": earnings_call_score if earnings_call_score is not None else 0,
                    "explanation": earnings_call_explanation
                },
                "industry_report_sentiment": {
                    "score": industry_report_score if industry_report_score is not None else 0,
                    "explanation": industry_report_explanation
                },
                "economic_report_sentiment": {
                    "score": economic_report_score if economic_report_score is not None else 0,
                    "explanation": economic_report_explanation
                }
            },
            "data_visualizations": {
                "latest_revenue": f"Q2: ${revenue:.1f}",
                "revenue_over_time": [
                    {"date": str(dates[-2]) if len(dates) > 1 else "N/A", "value": float(income_df['Revenue'].iloc[-2]) if len(income_df) > 1 else 0},
                    {"date": str(dates[-1]) if len(dates) > 0 else "N/A", "value": float(revenue)},
                ]
            },
            "final_recommendation": {
                "total_score": weighted_total_score,
                "recommendation": recommendation,
                "rationale": "No detailed rationale provided in code",
                "key_factors": ["Factor A", "Factor B"]
            },
            "company_info": {
                "sector": "Unknown",
                "c_suite": "CEO: N/A",
                "analysis": "No extra analysis here."
            }
        }

        # Add annual profit margins & industry margin to final analysis
        final_analysis["financial_analysis"]["annual_profit_margins"] = annual_profit_margins
        final_analysis["financial_analysis"]["industry_profit_margin"] = industry_profit_margin

        # Add the industry benchmarks
        final_analysis["financial_analysis"]["industry_benchmarks"] = final_industry_bench

        # Insert the optional quarterly data
        if "time_series_analysis" not in final_analysis["financial_analysis"]:
            final_analysis["financial_analysis"]["time_series_analysis"] = {}
        final_analysis["financial_analysis"]["time_series_analysis"]["quarterly"] = quarterly_data

        # Store the results for "demo_user"
        user_id = "demo_user"
        store_results_for_user(user_id, final_analysis)

        # Generate PDF (for internal use if needed)
        pdf_output = generate_pdf_report(
            financials=financials,
            ratios=ratios,
            cagr_values={
                'revenue_cagr': revenue_cagr,
                'net_income_cagr': net_income_cagr,
                'assets_cagr': assets_cagr,
                'liabilities_cagr': liabilities_cagr,
                'cashflow_cagr': cashflow_cagr
            },
            sentiment_results=sentiment_results,
            plots=plots,
            recommendation=recommendation,
            intrinsic_value_per_share=intrinsic_value_per_share,
            stock_price=stock_price,
            weighted_total_score=weighted_total_score,
            weights=weights,
            factor_scores=factor_scores,
            company_summary=company_summary,
            industry_summary=industry_summary,
            risks_summary=risks_summary,
            company_logo_path=file_paths.get('company_logo')
        )
        logger.info('Completed analyze_financials function successfully.')

        return redirect(url_for('dashboard'))

    except Exception as e:
        logger.exception('An unexpected error occurred during analysis.')
        return jsonify({'error': 'An unexpected error occurred.'}), 500








