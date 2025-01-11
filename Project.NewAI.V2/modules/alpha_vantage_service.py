import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()  # This will read variables from .env into os.environ

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

logger = logging.getLogger(__name__)

class AlphaVantageAPIError(Exception):
    """
    Custom exception raised when there's an error
    fetching or parsing data from Alpha Vantage.
    """
    pass

def fetch_income_statement(ticker: str) -> dict:
    """
    Fetches the income statement data from Alpha Vantage for a given ticker.
    Returns a dict with the raw JSON from the API response.
    Raises AlphaVantageAPIError if something goes wrong.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY
    }

    resp = requests.get(base_url, params=params)
    if resp.status_code != 200:
        raise AlphaVantageAPIError(
            f"Failed to fetch income statement. HTTP Status: {resp.status_code}"
        )

    data = resp.json()
    logger.debug(f"[fetch_income_statement] Raw response for {ticker}: {data}")
    if "annualReports" not in data:
        raise AlphaVantageAPIError(
            f"Invalid response from Alpha Vantage; 'annualReports' missing. "
            f"Full response: {data}"
        )

    return data

def fetch_balance_sheet(ticker: str) -> dict:
    """
    Fetches the balance sheet data from Alpha Vantage for a given ticker.
    Returns a dict with the raw JSON from the API response.
    Raises AlphaVantageAPIError if something goes wrong.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "BALANCE_SHEET",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY
    }

    resp = requests.get(base_url, params=params)
    if resp.status_code != 200:
        raise AlphaVantageAPIError(
            f"Failed to fetch balance sheet. HTTP Status: {resp.status_code}"
        )

    data = resp.json()
    logger.debug(f"[fetch_balance_sheet] Raw response for {ticker}: {data}")
    if "annualReports" not in data:
        raise AlphaVantageAPIError(
            f"Invalid response from Alpha Vantage; 'annualReports' missing. "
            f"Full response: {data}"
        )

    return data

def fetch_cash_flow(ticker: str) -> dict:
    """
    Fetches the cash flow statement data from Alpha Vantage for a given ticker.
    Returns a dict with the raw JSON from the API response.
    Raises AlphaVantageAPIError if something goes wrong.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "CASH_FLOW",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY
    }

    resp = requests.get(base_url, params=params)
    if resp.status_code != 200:
        raise AlphaVantageAPIError(
            f"Failed to fetch cash flow statement. HTTP Status: {resp.status_code}"
        )

    data = resp.json()
    logger.debug(f"[fetch_cash_flow] Raw response for {ticker}: {data}")
    if "annualReports" not in data:
        raise AlphaVantageAPIError(
            f"Invalid response from Alpha Vantage; 'annualReports' missing. "
            f"Full response: {data}"
        )

    return data

def fetch_all_statements(ticker: str) -> tuple:
    """
    Fetches the income statement, balance sheet, and cash flow statement
    from Alpha Vantage for the given ticker.
    Returns a tuple of three dicts: (income_data, balance_data, cash_flow_data).
    Raises AlphaVantageAPIError if something fails at any step.
    """
    income_data = fetch_income_statement(ticker)
    balance_data = fetch_balance_sheet(ticker)
    cash_flow_data = fetch_cash_flow(ticker)

    return (income_data, balance_data, cash_flow_data)

