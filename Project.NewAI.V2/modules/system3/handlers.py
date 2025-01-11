import os
import json
import logging
from logging.handlers import RotatingFileHandler
import traceback
import re
import requests  # ADDED for Alpha Vantage calls
from flask import request, jsonify, Blueprint
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pandas as pd
import difflib
from newsapi import NewsApiClient
from textblob import TextBlob
import numpy as np
import tiktoken
import openai

from smart_load_balancer import call_openai_smart, call_openai_embedding_smart
from model_selector import choose_model_for_task

from .def_model import DCFModel
import config
from modules.extensions import db
from config import (
    NEWSAPI_KEY,
    UPLOAD_FOLDER,
    SQLALCHEMY_DATABASE_URI
)

# We rely on alpha_vantage_service.py for statements:
from modules.alpha_vantage_service import (
    fetch_all_statements,
    AlphaVantageAPIError
)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )
    log_file = 'app.log'
    file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG)

system3_bp = Blueprint('system3_bp', __name__, template_folder='templates')

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

################################################################
# SQLAlchemy MODELS
################################################################
class AssumptionSet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sector = db.Column(db.String(50), nullable=False)
    industry = db.Column(db.String(50), nullable=False)
    sub_industry = db.Column(db.String(50), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    stock_ticker = db.Column(db.String(20), nullable=True)
    revenue_growth_rate = db.Column(db.Float, nullable=False)
    tax_rate = db.Column(db.Float, nullable=False)
    cogs_pct = db.Column(db.Float, nullable=False)
    wacc = db.Column(db.Float, nullable=False)
    terminal_growth_rate = db.Column(db.Float, nullable=False)
    operating_expenses_pct = db.Column(db.Float, nullable=False)
    feedbacks = db.relationship('Feedback', backref='assumption_set', lazy=True)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sector = db.Column(db.String(50), nullable=False)
    industry = db.Column(db.String(50), nullable=False)
    sub_industry = db.Column(db.String(50), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    revenue_growth_feedback = db.Column(db.String(20), nullable=True)
    tax_rate_feedback = db.Column(db.String(20), nullable=True)
    cogs_pct_feedback = db.Column(db.String(20), nullable=True)
    operating_expenses_feedback = db.Column(db.String(20), nullable=True)
    wacc_feedback = db.Column(db.String(20), nullable=True)

    assumption_set_id = db.Column(db.Integer, db.ForeignKey('assumption_set.id'), nullable=False)


################################################################
# FILE & DATA PROCESSING UTILS
################################################################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_json(json_like_str):
    cleaned = re.sub(r'//.*?\n', '\n', json_like_str)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    return cleaned

def parse_json_from_reply(reply):
    cleaned_reply = clean_json(reply)
    cleaned_reply = re.sub(r'"WACC"|\'WACC\'', '"wacc"', cleaned_reply, flags=re.IGNORECASE)
    json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.exception("JSON parsing failed in parse_json_from_reply")
            return {}
    return {}

def summarize_feedback(sector, industry, sub_industry, scenario):
    logger.debug(f"Summarizing feedback for {sector}, {industry}, {sub_industry}, {scenario}")
    try:
        feedback_entries = Feedback.query.join(AssumptionSet).filter(
            AssumptionSet.sector == sector,
            AssumptionSet.industry == industry,
            AssumptionSet.sub_industry == sub_industry,
            AssumptionSet.scenario == scenario
        ).all()

        if not feedback_entries:
            return "No relevant feedback available."

        assumption_feedback_counts = {
            'revenue_growth_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'tax_rate': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'cogs_pct': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'operating_expenses_pct': {'too_low': 0, 'about_right': 0, 'too_high': 0},
            'wacc': {'too_low': 0, 'about_right': 0, 'too_high': 0},
        }

        for entry in feedback_entries:
            for assumption in assumption_feedback_counts.keys():
                feedback_value = getattr(entry, f"{assumption}_feedback")
                if feedback_value in assumption_feedback_counts[assumption]:
                    assumption_feedback_counts[assumption][feedback_value] += 1

        summary_lines = []
        for assumption, counts in assumption_feedback_counts.items():
            total = sum(counts.values())
            if total > 0:
                summary = f"{assumption.replace('_', ' ').title()}:"
                for feedback_value, count in counts.items():
                    percentage = (count / total) * 100
                    summary += f" {percentage:.1f}% {feedback_value.replace('_', ' ')};"
                summary_lines.append(summary)

        comments = [f"- {entry.comments}" for entry in feedback_entries if entry.comments]
        if comments:
            summary_lines.append("\nUser Comments:")
            summary_lines.extend(comments)

        return "\n".join(summary_lines)
    except Exception as e:
        logger.exception("Error summarizing feedback")
        return "Error retrieving feedback."


def validate_assumptions(adjusted_assumptions):
    logger.debug(f"Validating assumptions: {adjusted_assumptions}")
    ranges = {
        'revenue_growth_rate': (0.0, 1.0, 0.05),
        'tax_rate': (0.01, 0.5, 0.21),
        'cogs_pct': (0.01, 1.0, 0.6),
        'wacc': (0.01, 0.2, 0.10),
        'terminal_growth_rate': (0.0, 0.05, 0.02),
        'operating_expenses_pct': (0.01, 1.0, 0.2)
    }
    validated_assumptions = {}
    for key, (min_val, max_val, default_val) in ranges.items():
        value = adjusted_assumptions.get(key)
        if value is not None:
            try:
                value = float(value)
                if value > 1.0:
                    value = value / 100.0
                value = max(min_val, min(value, max_val))
            except (ValueError, TypeError):
                value = default_val
        else:
            value = default_val
        validated_assumptions[key] = value
    logger.debug(f"Validated assumptions: {validated_assumptions}")
    return validated_assumptions

########
# Added
########

def guess_sector_industry_subindustry(ticker):
    """
    Calls GPT to classify the given ticker into sector, industry, sub-industry.
    Returns a dict, e.g.:
      {
        "sector": "Technology",
        "industry": "Software",
        "sub_industry": "Enterprise SaaS"
      }
    If GPT fails or returns no valid JSON, default to "Unknown".
    """
    if not ticker:
        return {
            "sector": "Unknown",
            "industry": "Unknown",
            "sub_industry": "Unknown"
        }

    # Build a prompt that instructs GPT to produce exactly that JSON structure
    prompt = f"""
We have a public company with ticker symbol {ticker}.
Please classify it into:
  - sector
  - industry
  - sub_industry
Return valid JSON with keys "sector", "industry", and "sub_industry".
"""
    assistant_reply = call_openai_api(prompt)
    cleaned = clean_json(assistant_reply)
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                # Ensure each key is present
                return {
                    "sector":  parsed.get("sector",  "Unknown"),
                    "industry": parsed.get("industry","Unknown"),
                    "sub_industry": parsed.get("sub_industry","Unknown")
                }
        except json.JSONDecodeError:
            pass
    return {
        "sector": "Unknown",
        "industry": "Unknown",
        "sub_industry": "Unknown"
}



################################################################
# GPT UTILS
################################################################
def call_openai_api(prompt):
    logger.debug(f"Calling OpenAI API with prompt[:1000]: {prompt[:1000]}...")
    try:
        messages = [
            {"role": "system", "content": "You are an expert financial analyst."},
            {"role": "user", "content": prompt}
        ]
        chosen_model = choose_model_for_task("complex_deep_analysis")
        response = call_openai_smart(
            messages=messages,
            model=chosen_model,
            temperature=0.2,
            max_tokens=750,
            max_retries=5
        )
        assistant_reply = response['choices'][0]['message']['content']
        logger.debug(f"Agent output: {assistant_reply[:500]}...")
        return assistant_reply
    except Exception as e:
        logger.exception("Error in OpenAI API call")
        return ""

def call_openai_api_with_messages(messages):
    logger.debug(f"Calling OpenAI API with messages: {messages}")
    try:
        chosen_model = choose_model_for_task("complex_deep_analysis")
        response = call_openai_smart(
            messages=messages,
            model=chosen_model,
            temperature=0.2,
            max_tokens=750,
            max_retries=5
        )
        assistant_reply = response['choices'][0]['message']['content']
        logger.debug(f"Assistant output: {assistant_reply[:500]}...")
        return assistant_reply
    except Exception as e:
        logger.exception("Error in OpenAI API call with messages")
        return ""

################################################################
# ALPHA VANTAGE PARSING
################################################################
def parse_alpha_vantage_json(income_json, balance_json, cash_flow_json):
    income_df = pd.DataFrame(income_json["annualReports"])
    balance_df = pd.DataFrame(balance_json["annualReports"])
    cash_flow_df = pd.DataFrame(cash_flow_json["annualReports"])

    for df in [income_df, balance_df, cash_flow_df]:
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

    return income_df, balance_df, cash_flow_df

################################################################
# MULTI-AGENT LOGIC
################################################################

def adjust_for_scenario(scenario):
    logger.debug(f"Adjusting for scenario: {scenario}")
    prompt = f"""
As a financial analyst, provide financial assumptions for a '{scenario}' scenario.
Return valid JSON:
{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}
"""
    assistant_reply = call_openai_api(prompt)
    c = clean_json(assistant_reply)
    c = re.sub(r'"WACC"|\'WACC\'', '"wacc"', c, flags=re.IGNORECASE)
    m = re.search(r'\{.*\}', c, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            return {}
    return {}

def adjust_for_sector(sector):
    if sector.lower()=="unknown":
        return {}
    prompt = f"""
As a financial analyst specializing in the {sector} sector,
return a JSON with:
{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}
"""
    rep = call_openai_api(prompt)
    c = clean_json(rep)
    c = re.sub(r'"WACC"|\'WACC\'', '"wacc"', c, flags=re.IGNORECASE)
    m = re.search(r'\{.*\}', c, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            return {}
    return {}

def adjust_for_industry(industry):
    if industry.lower()=="unknown":
        return {}
    prompt = f"""
As a financial analyst in the {industry} industry...
Return JSON for:
revenue_growth_rate, tax_rate, cogs_pct, wacc, terminal_growth_rate, operating_expenses_pct
"""
    r = call_openai_api(prompt)
    c = clean_json(r)
    c = re.sub(r'"WACC"|\'WACC\'', '"wacc"', c, flags=re.IGNORECASE)
    m = re.search(r'\{.*\}', c, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            return {}
    return {}

def adjust_for_sub_industry(sub_industry):
    if sub_industry.lower()=="unknown":
        return {}
    prompt = f"""
As a financial analyst in the {sub_industry} sub-industry...
Return JSON for revenue_growth_rate, tax_rate, cogs_pct, wacc, terminal_growth_rate, operating_expenses_pct
"""
    r = call_openai_api(prompt)
    c = clean_json(r)
    c = re.sub(r'"WACC"|\'WACC\'', '"wacc"', c, flags=re.IGNORECASE)
    m = re.search(r'\{.*\}', c, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            return {}
    return {}

def adjust_for_company(stock_ticker):
    """
    Dynamically fetches the company's overview from Alpha Vantage, then calls GPT
    with the name, industry, and a short business summary to produce company-specific assumptions.
    """
    if not stock_ticker:
        logger.warning("No stock ticker provided; cannot run company agent.")
        return {}

    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not ALPHAVANTAGE_API_KEY:
        logger.warning("Alpha Vantage API key not found. Returning empty from company agent.")
        return {}

    try:
        url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock_ticker}&apikey={ALPHAVANTAGE_API_KEY}"
        r = requests.get(url_overview, timeout=10)
        data_overview = r.json()

        if not data_overview or "Symbol" not in data_overview:
            logger.warning(f"No overview data available for {stock_ticker}. Using minimal fallback.")
            company_name = stock_ticker
            industry = "Unknown"
            business_summary = ""
        else:
            company_name = data_overview.get("Name", stock_ticker)
            industry = data_overview.get("Industry", "Unknown Industry")
            business_summary = data_overview.get("Description", "")

        prompt = f"""
We have {company_name} (ticker: {stock_ticker}), operating in the industry: {industry}.
Business Summary:
{business_summary}

As an expert financial analyst, propose recommended assumptions in JSON:
{{
  "revenue_growth_rate": ...,
  "tax_rate": ...,
  "cogs_pct": ...,
  "wacc": ...,
  "terminal_growth_rate": ...,
  "operating_expenses_pct": ...
}}
Use the above context to tailor these assumptions. No placeholders.
"""

        assistant_reply = call_openai_api(prompt)
        cleaned_reply = clean_json(assistant_reply)
        json_match = re.search(r'\{.*\}', cleaned_reply, re.DOTALL)
        if json_match:
            try:
                adjusted_assumptions = json.loads(json_match.group())
                if isinstance(adjusted_assumptions, dict):
                    return validate_assumptions(adjusted_assumptions)
                else:
                    logger.error("Company agent: returned data is not a dict.")
                    return {}
            except json.JSONDecodeError:
                logger.exception("Error parsing JSON in adjust_for_company")
                return {}
        else:
            logger.error("Failed to extract JSON from the company agent's LLM reply.")
            return {}
    except Exception as e:
        logger.exception(f"Error fetching or parsing company overview for {stock_ticker}: {e}")
        return {}

def adjust_based_on_feedback(sector, industry, sub_industry, scenario):
    fb = summarize_feedback(sector, industry, sub_industry, scenario)
    if "No relevant feedback" in fb:
        return {}
    prompt = f"""
User feedback summary: {fb}
Return JSON for:
revenue_growth_rate, tax_rate, cogs_pct, wacc, terminal_growth_rate, operating_expenses_pct
"""
    rep = call_openai_api(prompt)
    c = clean_json(rep)
    c = re.sub(r'"WACC"|\'WACC\'', '"wacc"', c, flags=re.IGNORECASE)
    m = re.search(r'\{.*\}', c, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            return {}
    return {}

def adjust_based_on_sentiment(stock_ticker):
    articles = newsapi.get_everything(q=stock_ticker, language='en', sort_by='relevancy', page_size=25)
    if 'articles' not in articles or len(articles['articles'])==0:
        return {}
    sentiments=[]
    for a in articles['articles']:
        content = a.get('content','')
        if content:
            tb = TextBlob(content)
            sentiments.append(tb.sentiment.polarity)
    if not sentiments:
        return {}
    avg = sum(sentiments)/len(sentiments)
    out={}
    if avg>0.1:
        out["revenue_growth_rate"] = 0.07
        out["wacc"] = 0.09
    elif avg<-0.1:
        out["revenue_growth_rate"] = 0.03
        out["wacc"] = 0.11
    else:
        out["revenue_growth_rate"] = 0.05
        out["wacc"] = 0.10
    return out

def adjust_based_on_historical_data(stock_ticker):
    """
    Fetches actual annualReports from Alpha Vantage, computes average revenue growth,
    tax rate, cogs%, opex%, and a simple WACC from Beta. All fully dynamic.
    """
    if not stock_ticker:
        logger.warning("No stock ticker provided to historical_data_agent.")
        return {}

    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not ALPHAVANTAGE_API_KEY:
        logger.warning("No alpha vantage API key. Returning empty from historical_data_agent.")
        return {}

    try:
        url_is = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={stock_ticker}&apikey={ALPHAVANTAGE_API_KEY}"
        resp_is = requests.get(url_is, timeout=10)
        data_is = resp_is.json()
        if "annualReports" not in data_is:
            logger.warning(f"Missing annualReports in income statement for {stock_ticker}.")
            return {}

        annual_reports = data_is["annualReports"]
        # Reverse to oldest->newest
        annual_reports.reverse()

        revenues, ibt_list, tax_list, cogs_list, sga_list = [], [], [], [], []
        for ar in annual_reports:
            rev = float(ar.get("totalRevenue", "0") or 0)
            ibt = float(ar.get("incomeBeforeTax", "0") or 0)
            tx  = float(ar.get("incomeTaxExpense", "0") or 0)
            cogs= float(ar.get("costOfRevenue", "0") or 0)
            sga = float(ar.get("sellingGeneralAdministrative", "0") or 0)
            revenues.append(rev)
            ibt_list.append(ibt)
            tax_list.append(tx)
            cogs_list.append(cogs)
            sga_list.append(sga)

        growths=[]
        for i in range(1,len(revenues)):
            if revenues[i-1]>0:
                g=(revenues[i]-revenues[i-1])/revenues[i-1]
                growths.append(g)
        avg_growth = np.mean(growths) if growths else 0.05

        valid_tax_rates=[]
        for ibt, tx in zip(ibt_list, tax_list):
            if ibt!=0:
                valid_tax_rates.append(tx/ibt)
        avg_tax = np.mean(valid_tax_rates) if valid_tax_rates else 0.21

        valid_cogs=[]
        for rv, cg in zip(revenues, cogs_list):
            if rv>0:
                valid_cogs.append(cg/rv)
        avg_cogs = np.mean(valid_cogs) if valid_cogs else 0.60

        valid_opex=[]
        for rv, sga in zip(revenues, sga_list):
            if rv>0:
                valid_opex.append(sga/rv)
        avg_opex = np.mean(valid_opex) if valid_opex else 0.20

        # fetch Beta from OVERVIEW
        url_ov = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock_ticker}&apikey={ALPHAVANTAGE_API_KEY}"
        ov_resp = requests.get(url_ov, timeout=10)
        data_ov = ov_resp.json()
        beta = 1.0
        if "Beta" in data_ov:
            try:
                b_val = float(data_ov["Beta"])
                beta = b_val if b_val>0 else 1.0
            except:
                pass
        risk_free_rate=0.02
        mrp=0.05
        cost_of_equity=risk_free_rate + beta*mrp
        average_wacc= cost_of_equity

        adjusted_assumptions={
            "revenue_growth_rate":avg_growth,
            "tax_rate":avg_tax,
            "cogs_pct":avg_cogs,
            "operating_expenses_pct":avg_opex,
            "wacc":average_wacc,
            "terminal_growth_rate":0.02
        }
        validated=validate_assumptions(adjusted_assumptions)
        return validated
    except Exception as e:
        logger.exception(f"Error in historical_data_agent for {stock_ticker}: {e}")
        return {}

################################################################
# Agent Weighting + Validation
################################################################

def get_agent_importance_weights():
    """
    Attempts to retrieve agent importance weights from GPT, ensuring
    each agent has a numeric (0..1) value. If GPT fails or gives partial
    data, we fallback to a default dictionary.
    """
    # We add explicit instructions: 
    # "No extra keys, and no text outside valid JSON" etc.
    prompt = """
You are a financial expert. Provide importance weights (0..1) in valid JSON, 
with exactly these keys (no extras):
{
  "scenario_agent": ...,
  "sector_agent": ...,
  "industry_agent": ...,
  "sub_industry_agent": ...,
  "company_agent": ...,
  "feedback_agent": ...,
  "sentiment_agent": ...,
  "historical_data_agent": ...,
  "user_agent": ...
}
The sum need not be exactly 1. We'll normalize afterwards.
No additional text. Valid JSON only.
"""

    rep = call_openai_api(prompt)
    parsed = parse_json_from_reply(rep)
    
    # Provide a fallback if GPT returns nothing or 
    # if it’s missing any required keys
    fallback = {
        "scenario_agent":          0.15,
        "sector_agent":            0.10,
        "industry_agent":          0.10,
        "sub_industry_agent":      0.10,
        "company_agent":           0.10,
        "feedback_agent":          0.05,
        "sentiment_agent":         0.05,
        "historical_data_agent":   0.10,
        "user_agent":              0.20
    }

    # If parsed is empty, or not a dict, or missing any required key
    if not isinstance(parsed, dict):
        logger.warning("GPT returned no dict for agent weights. Using fallback.")
        parsed = {}
    required_keys = set(fallback.keys())
    
    # 1) Merge fallback if any key is missing
    for rk in required_keys:
        if rk not in parsed:
            logger.warning(f"Missing agent weight for {rk}, using fallback.")
            parsed[rk] = fallback[rk]
    
    # 2) Convert to floats safely & clamp 0..1
    for k in list(parsed.keys()):
        try:
            val = float(parsed[k])
            # If GPT gave insane values
            if val < 0:
                val = 0
            elif val > 1:
                val = 1
            parsed[k] = val
        except:
            logger.warning(f"Non-numeric agent weight {k}, using fallback.")
            parsed[k] = fallback[k]

    # 3) Now we definitely have all 9 keys as floats in [0..1].
    #    Normalize them:
    s = sum(parsed.values())
    if s > 0:
        for k in parsed:
            parsed[k] = parsed[k] / s
    else:
        logger.warning("Sum of GPT importance weights is 0. Using fallback entirely.")
        parsed = fallback  # fallback has some valid distribution

    # optional debug:
    logger.debug(f"Final agent importance weights: {parsed}")

    return parsed

def validation_agent(
    final_adjustments, 
    sector, industry, sub_industry, 
    scenario, stock_ticker, 
    agent_adjustments
):
    """
    Calls GPT to get confidence scores for each agent’s output. If GPT fails
    or leaves any agent out, we default that agent's confidence to 1.0.
    """
    # We build a strict prompt so GPT must return exact JSON with 
    # numeric confidence scores for each agent. No extra text, no placeholders.
    prompt = f"""
You are a Validation Agent for {stock_ticker}, scenario={scenario}, sector={sector}, industry={industry}, sub_industry={sub_industry}.

We have agent outputs in JSON:
{json.dumps(agent_adjustments, indent=2)}

Return strictly valid JSON with two keys:
{{
  "reasoning": "...some textual explanation...",
  "confidence_scores": {{
    "scenario_agent": <float 0..1>,
    "sector_agent": <float 0..1>,
    "industry_agent": <float 0..1>,
    "sub_industry_agent": <float 0..1>,
    "company_agent": <float 0..1>,
    "feedback_agent": <float 0..1>,
    "sentiment_agent": <float 0..1>,
    "historical_data_agent": <float 0..1>,
    "user_agent": <float 0..1>
  }}
}}
No other text outside the JSON. 
Each confidence must be a float in [0,1].
"""

    raw_reply = call_openai_api(prompt)
    parsed = parse_json_from_reply(raw_reply)
    
    # Fallback to default=1.0 for each agent.
    fallback_scores = {
        "scenario_agent":         1.0,
        "sector_agent":           1.0,
        "industry_agent":         1.0,
        "sub_industry_agent":     1.0,
        "company_agent":          1.0,
        "feedback_agent":         1.0,
        "sentiment_agent":        1.0,
        "historical_data_agent":  1.0,
        "user_agent":             1.0
    }
    
    # If parse fails, we return "", fallback
    if not isinstance(parsed, dict):
        logger.warning("Validation agent: GPT returned no valid dict. Using fallback=1.0")
        return "", fallback_scores

    # 1) Extract reasoning if present
    reasoning = parsed.get("reasoning", "")

    # 2) Extract "confidence_scores" 
    cs = parsed.get("confidence_scores", {})
    if not isinstance(cs, dict):
        logger.warning("Validation agent: no dict for 'confidence_scores'. Using fallback=1.0")
        cs = {}

    # 3) Merge fallback for missing or invalid keys
    for agent_name, fallback_val in fallback_scores.items():
        if agent_name not in cs:
            logger.warning(f"Validation agent: missing confidence for {agent_name}, fallback=1.0")
            cs[agent_name] = fallback_val
        else:
            # Try casting. If invalid or out-of-range, fallback
            try:
                num = float(cs[agent_name])
                if num < 0.0:
                    num = 0.0
                elif num > 1.0:
                    num = 1.0
                cs[agent_name] = num
            except:
                logger.warning(f"Validation agent: non-numeric confidence for {agent_name}, fallback=1.0")
                cs[agent_name] = fallback_val

    # Optionally, we can log the final confidence dict
    logger.debug(f"Final validation confidence scores: {cs}")
    
    return reasoning, cs


def compute_final_adjustments_with_agents(agent_adjustments, agent_importance_weights, agent_confidence_scores):
    assumptions = [
        'revenue_growth_rate',
        'tax_rate',
        'cogs_pct',
        'wacc',
        'terminal_growth_rate',
        'operating_expenses_pct'
    ]
    final={}
    for a in assumptions:
        ws=0.0
        tw=0.0
        for agent_name,adjusts in agent_adjustments.items():
            if a in adjusts:
                val = adjusts[a]
                iw = agent_importance_weights.get(agent_name,0)
                cs = agent_confidence_scores.get(agent_name,1)
                w=iw*cs
                ws+= w* val
                tw+= w
        if tw>0:
            final[a] = ws/tw
        else:
            final[a] = 0
    return final

################################################################
# /base_case
################################################################
@system3_bp.route('/base_case', methods=['GET'])
def get_base_case():
    try:
        ticker = request.args.get('ticker','MSFT')
        scenario = "Neutral"  # base case scenario if none yet
        logger.info(f"base_case => ticker={ticker}, scenario={scenario}")

        # 1) fetch alpha vantage
        inc_json, bal_json, cf_json = fetch_all_statements(ticker)
        inc_df, bal_df, cf_df = parse_alpha_vantage_json(inc_json, bal_json, cf_json)
        if "totalRevenue" in inc_df.columns and len(inc_df)>0:
            latest_revenue = float(inc_df["totalRevenue"].iloc[0])
        else:
            latest_revenue = 1_000_000.0

        # 2) gather agent outputs (with GPT-based classification for sector/industry/sub_industry)
        classification = guess_sector_industry_subindustry(ticker)
        sector_str = classification.get("sector", "Unknown")
        industry_str = classification.get("industry", "Unknown")
        sub_industry_str = classification.get("sub_industry", "Unknown")

        scenario_out     = adjust_for_scenario(scenario)
        sector_out       = adjust_for_sector(sector_str)
        industry_out     = adjust_for_industry(industry_str)
        sub_industry_out = adjust_for_sub_industry(sub_industry_str)
        company_out      = adjust_for_company(ticker)
        feedback_out     = {}
        sentiment_out    = adjust_based_on_sentiment(ticker)
        historical_out   = adjust_based_on_historical_data(ticker)

        agent_adjustments = {
            "scenario_agent": scenario_out,
            "sector_agent": sector_out,
            "industry_agent": industry_out,
            "sub_industry_agent": sub_industry_out,
            "company_agent": company_out,
            "feedback_agent": feedback_out,
            "sentiment_agent": sentiment_out,
            "historical_data_agent": historical_out
        }

        # 3) weighting
        w = get_agent_importance_weights()
        reasoning, conf = validation_agent({}, sector_str, industry_str, sub_industry_str, scenario, ticker, agent_adjustments)
        merged = compute_final_adjustments_with_agents(agent_adjustments, w, conf)
        final_assumptions = validate_assumptions(merged)

        # 4) DCF
        dcf = DCFModel({"Revenue":latest_revenue}, final_assumptions)
        dcf.run_model()
        results = dcf.get_results()

        return jsonify({
            "intrinsic_value_per_share":results["intrinsic_value_per_share"],
            "final_assumptions":final_assumptions,
            "dcf_model_results":results,
            "scenario":scenario
        }),200
    except AlphaVantageAPIError as e:
        logger.exception("AlphaVantageAPIError: %s", e)
        return jsonify({"error":str(e)}),500
    except Exception as e:
        logger.exception("Error in get_base_case: %s", e)
        return jsonify({"error":str(e)}),500

################################################################
# /calculate_alpha
################################################################
@system3_bp.route('/calculate_alpha', methods=['POST'])
def calculate_custom_scenario():
    try:
        data = request.get_json()
        ticker = data.get("ticker","MSFT")
        scenario = data.get("scenario","Neutral")

        inc_json, bal_json, cf_json = fetch_all_statements(ticker)
        inc_df, bal_df, cf_df = parse_alpha_vantage_json(inc_json, bal_json, cf_json)
        if "totalRevenue" in inc_df.columns and len(inc_df)>0:
            latest_revenue = float(inc_df["totalRevenue"].iloc[0])
        else:
            latest_revenue = 1_000_000.0

        # classify sector/industry/subindustry
        classification = guess_sector_industry_subindustry(ticker)
        sector      = classification.get("sector", "Unknown")
        industry    = classification.get("industry","Unknown")
        sub_industry= classification.get("sub_industry","Unknown")

        scenario_out     = adjust_for_scenario(scenario)
        sector_out       = adjust_for_sector(sector)
        industry_out     = adjust_for_industry(industry)
        sub_industry_out = adjust_for_sub_industry(sub_industry)
        company_out      = adjust_for_company(ticker)
        feedback_out     = {}
        sentiment_out    = adjust_based_on_sentiment(ticker)
        historical_out   = adjust_based_on_historical_data(ticker)

        # user_agent
        user_agent = {}
        for k in ["revenue_growth_rate","tax_rate","cogs_pct","wacc","terminal_growth_rate","operating_expenses_pct"]:
            if k in data:
                user_agent[k] = float(data[k])

        agent_adjustments = {
            "scenario_agent": scenario_out,
            "sector_agent": sector_out,
            "industry_agent": industry_out,
            "sub_industry_agent": sub_industry_out,
            "company_agent": company_out,
            "feedback_agent": feedback_out,
            "sentiment_agent": sentiment_out,
            "historical_data_agent": historical_out,
            "user_agent": user_agent
        }

        w = get_agent_importance_weights()
        reasoning, conf = validation_agent({}, sector, industry, sub_industry, scenario, ticker, agent_adjustments)
        merged = compute_final_adjustments_with_agents(agent_adjustments, w, conf)
        final = validate_assumptions(merged)

        dcf = DCFModel({"Revenue":latest_revenue}, final)
        dcf.run_model()
        results = dcf.get_results()

        return jsonify({
            "intrinsic_value_per_share":results["intrinsic_value_per_share"],
            "final_assumptions":final,
            "scenario":scenario
        }),200
    except AlphaVantageAPIError as e:
        logger.exception("AlphaVantageAPIError: %s", e)
        return jsonify({"error":str(e)}),500
    except Exception as e:
        logger.exception("Error in calculate_custom_scenario: %s", e)
        return jsonify({"error":str(e)}),500

################################################################
# CSV-based /calculate
################################################################
def process_financial_csv(file_path, csv_name):
    try:
        df = pd.read_csv(file_path, index_col=0, thousands=',', quotechar='"')
        df.columns = df.columns.str.strip()
        df.index = df.index.str.strip()
        if 'ttm' in df.columns:
            df.drop(columns=['ttm'], inplace=True)
        df = df.transpose()
        df.reset_index(inplace=True)
        df.rename(columns={'index':'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df = df[df['Date'].notnull()]
        for col in df.columns:
            if col!='Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error processing {csv_name} CSV: {str(e)}")
        return None

def standardize_columns(df, column_mappings, csv_name):
    df_columns = df.columns.tolist()
    new_columns={}
    for standard_name, possible_names in column_mappings.items():
        for name in possible_names:
            if name in df_columns:
                new_columns[name]=standard_name
                break
    df.rename(columns=new_columns, inplace=True)
    unmapped_cols=[]
    for col in df_columns:
        if col not in new_columns.keys() and col in df.columns:
            unmapped_cols.append(col)
    return df, unmapped_cols

income_columns={
    "Revenue":["Revenue","TotalRevenue","Total Revenue","Sales"],
    "Net Income":["NetIncome","Net Income","Net Profit","Profit After Tax"],
    "Cost of Goods Sold (COGS)":["COGS","CostOfGoodsSold","Cost Of Goods Sold"],
    "Selling, General & Administrative (SG&A)":["SGA","SG&A","Selling Gen Admin"],
    "Depreciation & Amortization":["DepreciationAndAmortization","DepAndAmort","Depreciation & Amortization"],
    "Interest Expense":["InterestExpense","Interest Exp"],
    "Income Tax Expense":["IncomeTaxExpense","TaxExpense","Income Tax"]
}

balance_columns={
    "Total Assets":["TotalAssets","Total Assets"],
    "Total Liabilities":["TotalLiabilities","Total Liabilities","TotalLiabilitiesNetMinorityInterest"],
    "Shareholders Equity":["TotalEquity","Shareholders Equity","Total Equity","StockholdersEquity"],
    "Current Assets":["CurrentAssets","Current Assets"],
    "Current Liabilities":["CurrentLiabilities","Current Liabilities"],
    "CurrentDebt":["CurrentDebt","ShortTermDebt","Short-Term Debt","Current Debt"],
    "Long-Term Debt":["LongTermDebt","Long-Term Debt"],
    "Total Shares Outstanding":["SharesIssued","ShareIssued","Total Shares Outstanding","Total Shares","OrdinarySharesNumber"],
    "Inventory":["Inventory","Inventories"]
}

cashflow_columns={
    "Operating Cash Flow":["OperatingCashFlow","Operating Cash Flow","Cash from Operations"],
    "Capital Expenditures":["CapitalExpenditures","Capital Expenditures","CapEx","CapitalExpenditure","Capital Expenditure"]
}

@system3_bp.route('/calculate', methods=['POST'])
def calculate_scenario():
    """
    CSV-based route; merges multi-agent ensemble logic if you want.
    """
    import tempfile
    try:
        data=request.form.to_dict()
        scenario=data.get('scenario','Neutral')
        sector=data.get('sector','Unknown')
        industry=data.get('industry','Unknown')
        sub_industry=data.get('sub_industry','Unknown')
        stock_ticker=data.get('stock_ticker','AAPL')

        inc_file=request.files.get('income_statement')
        bal_file=request.files.get('balance_sheet')
        cf_file=request.files.get('cash_flow')
        if not inc_file or not bal_file or not cf_file:
            return jsonify({"error":"Missing CSVs"}),400

        with tempfile.NamedTemporaryFile(delete=False,suffix=".csv") as inc_f:
            inc_file.save(inc_f.name)
            inc_path=inc_f.name
        with tempfile.NamedTemporaryFile(delete=False,suffix=".csv") as bal_f:
            bal_file.save(bal_f.name)
            bal_path=bal_f.name
        with tempfile.NamedTemporaryFile(delete=False,suffix=".csv") as cf_f:
            cf_file.save(cf_f.name)
            cf_path=cf_f.name

        inc_df=process_financial_csv(inc_path,"income_statement")
        bal_df=process_financial_csv(bal_path,"balance_sheet")
        cf_df=process_financial_csv(cf_path,"cash_flow")

        os.remove(inc_path)
        os.remove(bal_path)
        os.remove(cf_path)

        if any(x is None for x in [inc_df,bal_df,cf_df]):
            return jsonify({"error":"Failed to parse CSVs."}),400

        inc_df, inc_unmapped=standardize_columns(inc_df,income_columns,"inc_csv")
        bal_df, bal_unmapped=standardize_columns(bal_df,balance_columns,"bal_csv")
        cf_df, cf_unmapped=standardize_columns(cf_df,cashflow_columns,"cf_csv")

        inc_df.sort_values('Date',inplace=True)
        if 'Revenue' in inc_df.columns and len(inc_df)>0:
            latest_revenue=float(inc_df['Revenue'].iloc[-1])
        else:
            latest_revenue=1_000_000.0

        scenario_out=adjust_for_scenario(scenario)
        sector_out=adjust_for_sector(sector)
        industry_out=adjust_for_industry(industry)
        sub_industry_out=adjust_for_sub_industry(sub_industry)
        company_out=adjust_for_company(stock_ticker)
        feedback_out={}
        sentiment_out={}
        historical_out=adjust_based_on_historical_data(stock_ticker)

        agent_adjustments={
            "scenario_agent":scenario_out,
            "sector_agent":sector_out,
            "industry_agent":industry_out,
            "sub_industry_agent":sub_industry_out,
            "company_agent":company_out,
            "feedback_agent":feedback_out,
            "sentiment_agent":sentiment_out,
            "historical_data_agent":historical_out
        }

        weights=get_agent_importance_weights()
        reasoning, conf=validation_agent({}, sector,industry,sub_industry, scenario,stock_ticker, agent_adjustments)
        merged=compute_final_adjustments_with_agents(agent_adjustments, weights, conf)
        final_assumptions=validate_assumptions(merged)

        dcf_model=DCFModel({"Revenue":latest_revenue},final_assumptions)
        dcf_model.run_model()
        results=dcf_model.get_results()

        return jsonify({
            "intrinsic_value_per_share":results["intrinsic_value_per_share"],
            "final_assumptions":final_assumptions,
            "scenario":scenario
        }),200
    except Exception as e:
        logger.exception("Error in CSV-based /calculate route: %s", e)
        return jsonify({"error":str(e)}),500
