# visuals.py

import pandas as pd

def generate_visual_data(income_df: pd.DataFrame, balance_df: pd.DataFrame, cashflow_df: pd.DataFrame) -> dict:
    """
    Build and return a dictionary that includes:
      - ttm_current (most recent 12-month total revenue)
      - ttm_previous (previous 12-month total revenue)
      - annual_revenue (list of the last 4 years of revenue)
      - yearly_data (detailed year-by-year breakdown for expanded charts)

    Parameters
    ----------
    income_df : pd.DataFrame
        Income statement data with columns like 'Revenue', 'Net Income', etc.
        Must have a 'Date' column of datetime type.
    balance_df : pd.DataFrame
        Balance sheet data with columns like 'Total Assets', 'Total Liabilities', etc.
        Must have a 'Date' column of datetime type.
    cashflow_df : pd.DataFrame
        Cash flow statement data with columns like 'Operating Cash Flow', etc.
        Must have a 'Date' column of datetime type.

    Returns
    -------
    dict
        A dictionary with keys matching what the front-end expects:
        {
            "ttm_current": float,   # total revenue of last 4 quarters
            "ttm_previous": float,  # total revenue of the 4 quarters before that
            "annual_revenue": [     # list of { "year": int, "value": float }
                { "year": 2019, "value": 1234567.0 },
                ...
            ],
            "yearly_data": [        # list of dicts, one per year, for expanded charts
                {
                    "year": int,
                    "revenue": float,
                    "net_income": float,
                    "cogs": float,
                    "sga": float,
                    "depreciation": float,
                    "interest_expense": float,
                    "income_tax": float,
                    "ebitda": float,
                    "ebit": float,
                    "ebt": float,
                    "assets": float,
                    "liabilities": float,
                    "operating_cash_flow": float
                },
                ...
            ]
        }
    """

    # Sort the income statement by Date in ascending order
    sorted_income = income_df.sort_values('Date', ascending=True)

    # 1) Calculate TTM (Trailing Twelve Months) for the most recent 4 quarters
    last_4_q = sorted_income.tail(4)
    if len(last_4_q) == 4:
        ttm_current_12mo = last_4_q['Revenue'].sum()
    else:
        ttm_current_12mo = 0.0

    # 2) Calculate TTM for the previous 4 quarters
    prev_8_q = sorted_income.tail(8)
    if len(prev_8_q) == 8:
        prev_4_q = prev_8_q.head(4)
        ttm_previous_12mo = prev_4_q['Revenue'].sum()
    else:
        ttm_previous_12mo = 0.0

    # 3) Build a list of the last 4 years of annual revenue
    all_years_sorted = sorted(income_df['Date'].dt.year.unique())
    if len(all_years_sorted) > 4:
        last_4_years = all_years_sorted[-4:]
    else:
        last_4_years = all_years_sorted

    annual_revenue_list = []
    for year in last_4_years:
        rows_this_year = income_df[income_df['Date'].dt.year == year]
        revenue_sum = rows_this_year['Revenue'].sum() if not rows_this_year.empty else 0.0
        annual_revenue_list.append({
            "year": int(year),
            "value": float(revenue_sum)
        })

    # 4) Build a detailed year-by-year breakdown for expanded charts
    yearly_data_list = []
    for year in last_4_years:
        # Filter rows for this year in each DataFrame
        sub_inc = income_df[income_df['Date'].dt.year == year]
        sub_bal = balance_df[balance_df['Date'].dt.year == year]
        sub_cf  = cashflow_df[cashflow_df['Date'].dt.year == year]

        # Sum or gather relevant fields
        rev_year = sub_inc['Revenue'].sum() if not sub_inc.empty else 0.0
        ni_year  = sub_inc['Net Income'].sum() if not sub_inc.empty else 0.0

        cogs = sub_inc['Cost of Goods Sold (COGS)'].sum() \
            if 'Cost of Goods Sold (COGS)' in sub_inc.columns else 0.0
        sga = sub_inc['Selling, General & Administrative (SG&A)'].sum() \
            if 'Selling, General & Administrative (SG&A)' in sub_inc.columns else 0.0
        da = sub_inc['Depreciation & Amortization'].sum() \
            if 'Depreciation & Amortization' in sub_inc.columns else 0.0
        int_exp = sub_inc['Interest Expense'].sum() \
            if 'Interest Expense' in sub_inc.columns else 0.0
        tax_exp = sub_inc['Income Tax Expense'].sum() \
            if 'Income Tax Expense' in sub_inc.columns else 0.0

        # Derived metrics
        ebitda_val = ni_year + int_exp + tax_exp + da
        ebit_val   = ni_year + int_exp + tax_exp
        ebt_val    = ni_year + tax_exp

        # Balance sheet (end-of-year snapshot)
        if not sub_bal.empty:
            bal_sorted = sub_bal.sort_values('Date', ascending=True)
            latest_bal = bal_sorted.iloc[-1]
            assets = float(latest_bal['Total Assets']) if 'Total Assets' in latest_bal else 0.0
            liab   = float(latest_bal['Total Liabilities']) if 'Total Liabilities' in latest_bal else 0.0
        else:
            assets, liab = 0.0, 0.0

        # Operating Cash Flow (sum or last row, depending on your preference)
        if not sub_cf.empty:
            ocf = sub_cf['Operating Cash Flow'].sum() if 'Operating Cash Flow' in sub_cf.columns else 0.0
        else:
            ocf = 0.0

        yearly_data_list.append({
            "year": int(year),
            "revenue": float(rev_year),
            "net_income": float(ni_year),
            "cogs": float(cogs),
            "sga": float(sga),
            "depreciation": float(da),
            "interest_expense": float(int_exp),
            "income_tax": float(tax_exp),
            "ebitda": float(ebitda_val),
            "ebit": float(ebit_val),
            "ebt": float(ebt_val),
            "assets": assets,
            "liabilities": liab,
            "operating_cash_flow": float(ocf)
        })

    # 5) Return a dictionary matching the front-end’s expectations
    data_visualizations = {
        "ttm_current": float(ttm_current_12mo),
        "ttm_previous": float(ttm_previous_12mo),
        "annual_revenue": annual_revenue_list,
        "yearly_data": yearly_data_list
    }

    return data_visualizations

