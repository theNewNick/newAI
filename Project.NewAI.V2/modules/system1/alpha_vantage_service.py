# modules/system1/alpha_vantage_service.py

import os
import requests
import datetime
import logging

# Load the Alpha Vantage API key from an environment variable.
# Make sure ALPHAVANTAGE_API_KEY is set in your environment or .env file.
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')

def get_annual_price_change(symbol):
    """
    Fetches daily adjusted stock data from Alpha Vantage, 
    identifies the most recent trading day, then finds the 
    closing price approximately 12 months prior. Returns a
    tuple (current_price, pct_change_12mo).

    If data is unavailable or an error occurs, it returns:
      - (None, None) if neither current nor historical price could be found.
      - (current_price, None) if only the historical price is missing.
    
    Make sure to handle the scenario when there's no data 
    or the symbol is invalid.
    """
    # Alpha Vantage endpoint
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",  # ensure we get at least one year of data
        "apikey": ALPHAVANTAGE_API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        # The time series data is expected under the "Time Series (Daily)" key.
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            logging.warning(
                f"No daily time series data found for symbol='{symbol}'. "
                f"Response may indicate an invalid symbol or API limit issue."
            )
            return (None, None)

        # Sort the dates (keys) in ascending order: earliest -> latest
        all_dates = sorted(time_series.keys())
        # The last element should correspond to the most recent trading day.
        most_recent_date = all_dates[-1]
        current_price_str = time_series[most_recent_date].get("4. close")
        if not current_price_str:
            logging.warning(
                f"No '4. close' field found for the most recent date: '{most_recent_date}' "
                f"on symbol='{symbol}'."
            )
            return (None, None)

        current_price = float(current_price_str)

        # Convert the most recent date to a datetime object
        recent_date_obj = datetime.datetime.strptime(most_recent_date, "%Y-%m-%d")
        # Approximate 12 months in the past
        one_year_ago = recent_date_obj - datetime.timedelta(days=365)

        one_year_ago_price = None
        check_date = one_year_ago
        boundary_date = datetime.datetime(2000, 1, 1)  # just a safety lower bound

        # Loop backward day by day from one_year_ago until we find a trading day
        while check_date >= boundary_date:
            candidate_date_str = check_date.strftime("%Y-%m-%d")
            if candidate_date_str in time_series:
                close_str = time_series[candidate_date_str].get("4. close")
                if close_str:
                    one_year_ago_price = float(close_str)
                break
            # step back one day
            check_date -= datetime.timedelta(days=1)

        if not one_year_ago_price:
            # We found a current price but no historical price
            return (current_price, None)

        # Compute the percentage change
        pct_change_12mo = ((current_price - one_year_ago_price) / one_year_ago_price) * 100.0

        return (current_price, pct_change_12mo)

    except Exception as e:
        logging.error(f"Error fetching data from Alpha Vantage: {e}", exc_info=True)
        return (None, None)

