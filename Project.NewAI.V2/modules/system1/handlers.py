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

# 1) We ensure session is imported
from flask import session

import PyPDF2
from flask import request, send_file, jsonify, Blueprint, redirect, url_for
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
from asgiref.wsgi import WsgiToAsgi
import config

# Additional imports for S3 uploading
import boto3
import uuid

# Define the blueprint
system1_bp = Blueprint('system1_bp', __name__, template_folder='templates')

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API key configuration
openai.api_key = config.OPENAI_API_KEY

UPLOAD_FOLDER = config.UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    'csv': {'csv'},
    'pdf': {'pdf'},
    'image': {'png', 'jpg', 'jpeg', 'gif'}
}

# S3 configuration
S3_BUCKET_NAME = config.S3_BUCKET_NAME
s3_client = boto3.client('s3', region_name=config.AWS_REGION)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def dcf_analysis(projected_free_cash_flows, wacc, terminal_value, projection_years):
    logger.debug('Starting DCF analysis.')
    discounted_free_cash_flows = [
        fcf / (1 + wacc) ** i for i, fcf in enumerate(projected_free_cash_flows, 1)
    ]
    discounted_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
    dcf_value = sum(discounted_free_cash_flows) + discounted_terminal_value
    logger.debug('Completed DCF analysis.')
    return dcf_value

def safe_divide(numerator, denominator):
    try:
        result = numerator / denominator if denominator != 0 else None
        logger.debug(f'Safe divide {numerator} / {denominator} = {result}')
        return result
    except Exception as e:
        logger.error(f'Error in safe_divide: {str(e)}')
        return None

def calculate_ratios(financials, benchmarks):
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
            response = await openai.ChatCompletion.acreate(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                max_tokens=500,
                n=1,
                temperature=0.5,
            )
            summary = response.choices[0].message.content.strip()
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

async def generate_summaries_and_risks(ten_k_text):
    logger.debug('Generating company, industry summaries, and risk considerations from 10-K report.')

    company_prompt = f"""
Summarize the company based on the following 10-K report:
{ten_k_text}
"""
    industry_prompt = f"""
Summarize the industry based on the following 10-K report:
{ten_k_text}
"""
    risks_prompt = f"""
Identify and summarize the key risk considerations from the 10-K report:
{ten_k_text}
"""

    async def summarize_with_token_check(prompt, max_tokens=500):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        max_context_length = 7000

        tokens = encoding.encode(prompt)
        if len(tokens) > max_context_length - max_tokens:
            logger.warning('Prompt too long, chunking...')
            chunk_size = max_context_length - max_tokens
            chunks = [encoding.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
            summaries = []
            for chunk in chunks:
                summary = await call_openai_summarization(chunk)
                if summary:
                    summaries.append(summary)
            return " ".join(summaries)
        return await call_openai_summarization(prompt)

    company_summary_task = summarize_with_token_check(company_prompt)
    industry_summary_task = summarize_with_token_check(industry_prompt)
    risks_task = summarize_with_token_check(risks_prompt)
    company_summary, industry_summary, risks_summary = await asyncio.gather(
        company_summary_task, industry_summary_task, risks_task
    )

    logger.debug('Summaries generated for 10-K data.')
    return company_summary, industry_summary, risks_summary

async def call_openai_analyze_sentiment(text, context):
    retry_delay = 5
    max_retries = 5
    prompt = f"""
As an expert financial analyst, analyze the following {context}.
Provide a sentiment score between -1 (very negative) and 1 (very positive).
Also, briefly explain the main factors.

Text:
{text}

Response Format:
Sentiment Score: [score]
Explanation: [brief explanation]
"""
    for attempt in range(max_retries):
        try:
            logger.debug(f'Attempting sentiment analysis, attempt {attempt+1}.')
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in sentiment analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                temperature=0.5,
            )
            content = response.choices[0].message.content.strip()
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
    logger.error('Failed sentiment analysis after multiple attempts.')
    return None, ""

async def summarize_text_async(text):
    logger.debug('Starting text summarization.')
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

async def process_documents(earnings_call_text, industry_report_text, economic_report_text):
    logger.debug('Summarizing extracted texts for earnings, industry, and economic reports.')
    summaries = await asyncio.gather(
        summarize_text_async(earnings_call_text),
        summarize_text_async(industry_report_text),
        summarize_text_async(economic_report_text)
    )

    if not all(summaries):
        logger.warning('Error summarizing one or more documents.')
        return None

    earnings_call_summary, industry_report_summary, economic_report_summary = summaries
    logger.debug('Analyzing sentiments on the summarized texts.')
    sentiments = await asyncio.gather(
        call_openai_analyze_sentiment(earnings_call_summary, "earnings call transcript"),
        call_openai_analyze_sentiment(industry_report_summary, "industry report"),
        call_openai_analyze_sentiment(economic_report_summary, "economic report")
    )
    return sentiments

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
        logger.debug(f"{csv_name} DataFrame:\n{df.head()}")
        logger.debug(f'Completed processing CSV: {file_path}')
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

from .pdf_generation import generate_pdf_report

@system1_bp.route('/analyze', methods=['POST'])
def analyze_financials():
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

        logger.debug('Extracting text from the 10-K report.')
        ten_k_text = extract_text_from_pdf(file_paths['ten_k_report'])
        if not ten_k_text:
            error_message = 'Error extracting text from the 10-K report.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400
        else:
            logger.info('Extracted text from the 10-K successfully.')

        # Generate real summaries from the 10-K
        company_summary, industry_summary, risks_summary = run_async_function(
            generate_summaries_and_risks(ten_k_text)
        )
        logger.info('Generated 10-K summaries successfully.')

        # Retrieve form inputs
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

        company_name = request.form.get('company_name', 'N/A')
        financials = {'company_name': company_name}
        logger.info(f"Company Name: {company_name}")

        income_df = process_financial_csv(file_paths['income_statement'], 'income_statement')
        balance_df = process_financial_csv(file_paths['balance_sheet'], 'balance_sheet')
        cashflow_df = process_financial_csv(file_paths['cash_flow'], 'cash_flow')

        if income_df is None or balance_df is None or cashflow_df is None:
            error_message = 'Error processing CSV files.'
            logger.error(error_message)
            return jsonify({'error': error_message}), 400

        # Standardize columns
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

        for df, name in [(income_df, 'income_statement'), (balance_df, 'balance_sheet'), (cashflow_df, 'cash_flow')]:
            if 'Date' not in df.columns:
                error_message = f"'Date' column missing in {name} CSV."
                logger.error(error_message)
                return jsonify({'error': error_message}), 400

        # Sorting data
        income_df.sort_values('Date', inplace=True)
        balance_df.sort_values('Date', inplace=True)
        cashflow_df.sort_values('Date', inplace=True)

        income_df.reset_index(drop=True, inplace=True)
        balance_df.reset_index(drop=True, inplace=True)
        cashflow_df.reset_index(drop=True, inplace=True)

        # Convert columns to numeric
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

        # Extract final row for analysis
        latest_date = balance_df['Date'].max()
        long_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'Long-Term Debt'].values[0] if 'Long-Term Debt' in balance_df.columns else 0
        short_term_debt = balance_df.loc[balance_df['Date'] == latest_date, 'CurrentDebt'].values[0] if 'CurrentDebt' in balance_df.columns else 0
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
        projected_growth_rate = min(average_growth_rate, growth_rate)
        logger.debug(f'Projected growth rate: {projected_growth_rate}')

        projection_years = 5
        last_free_cash_flow = free_cash_flows.iloc[-1]
        projected_fcfs = [
            last_free_cash_flow * (1 + projected_growth_rate) ** i
            for i in range(1, projection_years + 1)
        ]

        if wacc <= growth_rate:
            error_message = 'WACC must be greater than the growth rate for DCF calculation.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        terminal_value = projected_fcfs[-1] * (1 + growth_rate) / (wacc - growth_rate)
        dcf_total = dcf_analysis(projected_fcfs, wacc, terminal_value, projection_years)
        intrinsic_value_per_share = safe_divide(dcf_total, total_shares_outstanding)
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

        financials.update({
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
        })
        logger.info('Extracted financial data successfully.')

        # Ratios
        ratios = calculate_ratios(financials, benchmarks)
        factor2_score = ratios['Normalized Factor 2 Score']

        # Extract PDF text for earnings/industry/economic
        earnings_call_text = extract_text_from_pdf(file_paths['earnings_call'])
        industry_report_text = extract_text_from_pdf(file_paths['industry_report'])
        economic_report_text = extract_text_from_pdf(file_paths['economic_report'])

        if not all([earnings_call_text, industry_report_text, economic_report_text]):
            error_message = 'Error extracting text from one or more PDF files.'
            logger.warning(error_message)
            return jsonify({'error': error_message}), 400

        # Run async tasks for sentiment
        sentiments = run_async_function(
            process_documents(earnings_call_text, industry_report_text, economic_report_text)
        )
        if sentiments is None:
            error_message = 'An error occurred during document processing.'
            logger.error(error_message)
            return jsonify({'error': error_message}), 500

        (earnings_call_score, earnings_call_explanation), \
        (industry_report_score, industry_report_explanation), \
        (economic_report_score, economic_report_explanation) = sentiments

        def map_sentiment_to_score(s):
            if s is None:
                return 0
            elif s > 0.5:
                return 1
            elif s < -0.5:
                return -1
            else:
                return 0

        factor4_score = map_sentiment_to_score(earnings_call_score)
        factor5_score = map_sentiment_to_score(industry_report_score)
        factor6_score = map_sentiment_to_score(economic_report_score)

        # Check time-series length
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
        plots = {}
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

        # We'll store all final data in a dictionary, then put it in session
        analysis_data = {
            "company_summary": company_summary or "",
            "industry_summary": industry_summary or "",
            "risks_summary": risks_summary or "",
            "dcf_intrinsic_value": intrinsic_value_per_share if intrinsic_value_per_share else 0,
            "recommendation": recommendation,
            "weighted_total_score": weighted_total_score,
            "ratios": ratios,
            "sentiment_results": sentiment_results,
            "factor_scores": factor_scores,
        }

        # Put your final data in session
        session['analysis_result'] = analysis_data
        logger.info("Analysis results saved in session for the dashboard to retrieve.")

        # Now generate PDF for optional download
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
            company_summary=analysis_data["company_summary"],
            industry_summary=analysis_data["industry_summary"],
            risks_summary=analysis_data["risks_summary"],
            company_logo_path=file_paths.get('company_logo')
        )

        logger.info('Completed analyze_financials function successfully.')

        # Upload PDF to S3 (optional)
        report_id = str(uuid.uuid4())
        pdf_filename = f"analysis_report_{report_id}.pdf"

        try:
            pdf_output.seek(0)
            s3_client.upload_fileobj(
                pdf_output,
                S3_BUCKET_NAME,
                pdf_filename,
                ExtraArgs={"ContentType": "application/pdf"}
            )
            logger.info(f"PDF uploaded to S3 with key: {pdf_filename}")
        except Exception as e:
            logger.error(f"Error uploading PDF to S3: {e}")
            return jsonify({'error': 'Failed to store PDF in S3.'}), 500

        # Finally, redirect to the dashboard
        return redirect(url_for('dashboard'))

    except Exception as e:
        logger.exception('An unexpected error occurred during analysis.')
        return jsonify({'error': 'An unexpected error occurred.'}), 500

#
# Updated endpoints to read from session instead of returning static placeholders
#

@system1_bp.route('/company_report_data', methods=['GET'])
def company_report_data():
    analysis = session.get('analysis_result')
    if not analysis:
        # If the user hasn't run the analysis yet
        return jsonify({
            "executive_summary": "No analysis found in session.",
            "company_summary": "",
            "industry_summary": "",
            "risk_considerations": ""
        })

    return jsonify({
        "executive_summary": "Short summary if you have one (hard-coded or from your text).",
        "company_summary": analysis.get("company_summary", ""),
        "industry_summary": analysis.get("industry_summary", ""),
        "risk_considerations": analysis.get("risks_summary", "")
    })


@system1_bp.route('/financial_analysis_data', methods=['GET'])
def financial_analysis_data():
    analysis = session.get('analysis_result')
    if not analysis:
        return jsonify({
            "dcf_intrinsic_value": 0,
            "ratios": {},
            "time_series_analysis": {}
        })

    # You can expand time_series_analysis if you have CAGR data etc. stored
    return jsonify({
        "dcf_intrinsic_value": analysis.get("dcf_intrinsic_value", 0),
        "ratios": analysis.get("ratios", {}),
        "time_series_analysis": {
            "Revenue CAGR": "N/A",
            "Net Income CAGR": "N/A",
            "Assets CAGR": "N/A",
            "Liabilities CAGR": "N/A",
            "Operating Cash Flow CAGR": "N/A"
        }
    })


@system1_bp.route('/sentiment_data', methods=['GET'])
def sentiment_data():
    analysis = session.get('analysis_result')
    if not analysis:
        return jsonify({
            "earnings_call_sentiment": {"score": 0, "explanation": "No analysis found"},
            "industry_report_sentiment": {"score": 0, "explanation": ""},
            "economic_report_sentiment": {"score": 0, "explanation": ""}
        })

    sr = analysis.get("sentiment_results", [])
    e_call = next((x for x in sr if x['title'] == 'Earnings Call Sentiment'), None)
    i_repo = next((x for x in sr if x['title'] == 'Industry Report Sentiment'), None)
    econ_repo = next((x for x in sr if x['title'] == 'Economic Report Sentiment'), None)

    return jsonify({
        "earnings_call_sentiment": {
            "score": e_call['score'] if e_call else 0,
            "explanation": e_call['explanation'] if e_call else ''
        },
        "industry_report_sentiment": {
            "score": i_repo['score'] if i_repo else 0,
            "explanation": i_repo['explanation'] if i_repo else ''
        },
        "economic_report_sentiment": {
            "score": econ_repo['score'] if econ_repo else 0,
            "explanation": econ_repo['explanation'] if econ_repo else ''
        }
    })


@system1_bp.route('/data_visualizations_data', methods=['GET'])
def data_visualizations_data():
    analysis = session.get('analysis_result')
    if not analysis:
        return jsonify({"error": "No analysis data in session. Please run analysis first."}), 400

    # You can store more complex time-series or chart data in the session if you want
    return jsonify({
        "benchmark_comparison": {
            "Ratios": ["Debt-to-Equity Ratio", "Current Ratio", "P/E Ratio", "P/B Ratio"],
            "Company": [],
            "Industry": []
        },
        "revenue_over_time": [],
        "net_income_over_time": [],
        "operating_cash_flow_over_time": [],
        "assets_liabilities_over_time": {}
    })


@system1_bp.route('/final_recommendation', methods=['GET'])
def final_recommendation():
    analysis = session.get('analysis_result')
    if not analysis:
        return jsonify({
            "total_score": 0,
            "recommendation": "None"
        })

    return jsonify({
        "total_score": analysis.get("weighted_total_score", 0),
        "recommendation": analysis.get("recommendation", "None")
    })


@system1_bp.route('/company_info_data', methods=['GET'])
def company_info_data():
    # Possibly also dynamic; for now you can keep placeholders or fill from analysis
    return jsonify({
        "c_suite_executives": "John Doe (CEO), Jane Smith (CFO), Alex Johnson (CTO)",
        "shares_outstanding": 1500000000,
        "wacc": 0.10,
        "pe_ratio": 22.4,
        "ps_ratio": 5.1,
        "sector": "Technology",
        "industry": "Semiconductors",
        "sub_industry": "Integrated Circuits"
    })


@system1_bp.route('/get_report', methods=['GET'])
def get_report():
    pdf_key = 'analysis_report_<your-id>.pdf'  # or read from session, DB, etc.
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(S3_BUCKET_NAME, pdf_key, buffer)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='financial_report.pdf'
        )
    except Exception as e:
        logger.error(f"Error fetching PDF: {e}")
        return "Report not found", 404
