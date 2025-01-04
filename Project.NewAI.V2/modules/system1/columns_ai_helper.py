# columns_ai_helper.py

import logging
logger = logging.getLogger(__name__)

# Suppose you already have a function call_chatgpt in your project.
# We'll make a placeholder for demonstration:
def call_chatgpt(prompt):
    """
    Placeholder function that sends 'prompt' to ChatGPT (or any LLM)
    and returns the response as a string.
    You need to implement this based on how your environment calls GPT.
    """
    # For now, let's just pretend it always says 'None'.
    # In reality, you'd integrate your actual GPT API call here.
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
    Queries ChatGPT to guess which standard expense column `column_name` might map to.
    Returns a string that is exactly one of ALLOWED_STANDARD_COLUMNS if there's a match,
    or 'None' if ChatGPT can't match it.
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
