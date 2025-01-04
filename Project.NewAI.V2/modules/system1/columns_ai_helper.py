# columns_ai_helper.py

import logging
logger = logging.getLogger(__name__)

import openai  # Make sure you've installed openai >= 0.27.0

def call_chatgpt(prompt: str) -> str:
    """
    Calls OpenAI ChatCompletion with the given prompt and returns the response as a string.
    Make sure OPENAI_API_KEY is set in your environment or passed in some other way.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # Keep it deterministic if you want exact matches
            max_tokens=50      # Enough room to return a short leftover column name
        )
        answer = response["choices"][0]["message"]["content"]
        return answer.strip()
    except Exception as e:
        logger.warning(f"Error calling GPT: {e}")
        return "None"


ALLOWED_STANDARD_COLUMNS = [
    "Cost of Goods Sold (COGS)",
    "Selling, General & Administrative (SG&A)",
    "Depreciation & Amortization",
    "Interest Expense",
    "Income Tax Expense"
]


def guess_column_meaning(column_name: str) -> str:
    """
    Queries ChatGPT to guess which standard expense column `column_name`
    might map to, using ALLOWED_STANDARD_COLUMNS as a reference.

    Returns:
    --------
    str
        Exactly one of ALLOWED_STANDARD_COLUMNS if there's a match,
        or 'None' if GPT can't match it.

    Example usage:
        guess = guess_column_meaning("CostOfRevenue")
        # might return "Cost of Goods Sold (COGS)" or "None".
    """
    prompt = f"""
We have a column in an income statement named: '{column_name}'.
Possible categories: {ALLOWED_STANDARD_COLUMNS + ['None']}

Which one best fits the column name, or 'None' if no match?
Respond EXACTLY with the matched category or 'None'.
"""
    try:
        response = call_chatgpt(prompt)
        guess = response.strip()
        if guess in ALLOWED_STANDARD_COLUMNS:
            return guess
        else:
            return 'None'
    except Exception as e:
        logger.warning(f"Error calling ChatGPT for column '{column_name}': {str(e)}")
        return 'None'


def guess_best_match_from_list(needed_col: str, leftover_cols: list, csv_name: str) -> str:
    """
    Calls GPT with a prompt listing leftover_cols, telling GPT we want
    the SINGLE best match for 'needed_col' in a specific CSV (e.g., 'income_statement').

    It returns exactly one leftover column name from leftover_cols
    or 'None' if GPT can't find a reasonable match.

    Parameters:
    -----------
    needed_col : str
        The standard column name you still need (e.g. "Cost of Goods Sold (COGS)").
    leftover_cols : list of str
        The list of unmapped columns from your DataFrame after initial standardization.
    csv_name : str
        A short descriptor like "income_statement", "balance_sheet", etc.

    Returns:
    --------
    str
        If GPT successfully picks one column name from leftover_cols, return it verbatim.
        Otherwise, return 'None'.

    Example usage:
        leftover = ["CostOfRevenue", "GrossProfit", "OperatingExpense"]
        guess = guess_best_match_from_list("Cost of Goods Sold (COGS)", leftover, "income_statement")
        # might return "CostOfRevenue" or "None".

    Note:
        This function relies on call_chatgpt() returning a single string.
        Ensure your GPT prompt enforces a strict response (only one leftover name or 'None').
    """
    if not leftover_cols:
        return "None"

    prompt = f"""
We have a {csv_name}, and we need the column that represents '{needed_col}'.

Here is a list of leftover column names:
{leftover_cols}

Which ONE of these leftover names is the best match for '{needed_col}'?
Return EXACTLY one name (verbatim), or 'None' if there's no reasonable match.

No extra words, no quotes, no punctuation—just one leftover column name or 'None'.
"""
    try:
        response = call_chatgpt(prompt)
        guess = response.strip()
        # Only accept guess if it's literally one of leftover_cols
        if guess in leftover_cols:
            return guess
        else:
            return "None"
    except Exception as e:
        logger.warning(f"Error calling GPT to match needed '{needed_col}' in {csv_name}: {str(e)}")
        return "None"
