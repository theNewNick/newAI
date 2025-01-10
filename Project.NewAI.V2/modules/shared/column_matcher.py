# modules/shared/column_matcher.py

import difflib
import logging
import re
import json

from model_selector import choose_model_for_task
from smart_load_balancer import call_openai_smart

logger = logging.getLogger(__name__)


###############################################################################
# EXPANDED KNOWN PATTERNS
###############################################################################
# For improved direct matching coverage, you can store or extend a bigger dict:
# Each standardized key has a list of synonyms or variations.
# If you like, you can maintain multiple sets (income, balance, cashflow)
# in one place, or unify them. Here we keep just one combined example,
# since you mentioned you'd prefer everything in one file.
# Feel free to expand this further as needed.

EXPANDED_KNOWN_PATTERNS = {
    "Revenue": [
        "Revenue",
        "Sales",
        "TotalRevenue",
        "OperatingRevenue",
        "SalesRevenue",
        "NetSales",
        "GrossRevenue"
    ],
    "Net Income": [
        "NetIncome",
        "Net Profit",
        "Net Earnings",
        "ProfitAfterTax",
        "Net Income",
        "EarningsAfterTax"
    ],
    "Cost of Goods Sold (COGS)": [
        "COGS",
        "CostOfGoodsSold",
        "CostOfRevenue",
        "Cost of Goods Sold",
        "Cost of Revenue"
    ],
    "Selling, General & Administrative (SG&A)": [
        "SG&A",
        "Selling General & Administrative",
        "Selling, General and Administrative Expenses",
        "OperatingExpenses",
        "Operating Expenses"
    ],
    "Depreciation & Amortization": [
        "DepreciationAndAmortization",
        "Depreciation & Amortization",
        "DepAndAmort",
        "DepreciationAmortizationDepletion",
        "AmortizationOfIntangibles"
    ],
    "Interest Expense": [
        "InterestExpense",
        "Interest Exp",
        "Interest_Paid",
        "InterestPayment"
    ],
    "Income Tax Expense": [
        "IncomeTaxExpense",
        "TaxExpense",
        "Income Tax",
        "CorporateTax",
        "TaxesPaid"
    ],
    "Total Assets": [
        "TotalAssets",
        "Total Assets"
    ],
    "Total Liabilities": [
        "TotalLiabilities",
        "Total Liabilities",
        "TotalLiabilitiesNetMinorityInterest"
    ],
    "Shareholders Equity": [
        "TotalEquity",
        "Shareholders Equity",
        "Total Equity",
        "StockholdersEquity"
    ],
    "Current Assets": [
        "CurrentAssets",
        "Current Assets",
        "TotalCurrentAssets"
    ],
    "Current Liabilities": [
        "CurrentLiabilities",
        "Current Liabilities",
        "TotalCurrentLiabilities"
    ],
    "CurrentDebt": [
        "CurrentDebt",
        "ShortTermDebt",
        "Short-Term Debt",
        "Current Debt"
    ],
    "Long-Term Debt": [
        "LongTermDebt",
        "Long-Term Debt",
        "Non-Current Debt"
    ],
    "Total Shares Outstanding": [
        "SharesIssued",
        "ShareIssued",
        "Total Shares Outstanding",
        "Total Shares",
        "OrdinarySharesNumber"
    ],
    "Inventory": [
        "Inventory",
        "Inventories"
    ],
    "Operating Cash Flow": [
        "OperatingCashFlow",
        "Operating Cash Flow",
        "CashFromOperations",
        "CashFlowFromOperatingActivities",
        "CashFlowFromContinuingOperatingActivities"
    ],
    "Capital Expenditures": [
        "CapitalExpenditures",
        "Capital Expenditures",
        "CapEx",
        "CapitalExpenditure",
        "PurchaseOfPPE",
        "SaleOfPPE"
    ],
}


def match_columns(
    incoming_cols,
    known_patterns,
    partial_match_threshold=0.7,
    do_second_pass=False
):
    """
    Performs a three-step matching process on a list of column names:
      1) Direct matching against known synonyms
      2) Partial/fuzzy matching via difflib.get_close_matches()
      3) GPT-based fallback for any columns still unmatched

    Additionally, if do_second_pass=True, we perform a final "review"
    to see if GPT wants to adjust any mapping (a simple voting approach).

    Parameters
    ----------
    incoming_cols : list of str
        Column names to match.
    known_patterns : dict of str -> list of str
        A dictionary mapping a standardized name to a list of accepted synonyms.
    partial_match_threshold : float
        Cutoff for fuzzy matching. Higher values require closer similarity.
    do_second_pass : bool
        If True, performs a final GPT "voting" pass for the entire mapping.

    Returns
    -------
    dict
        Maps each column in `incoming_cols` to a standardized name or None.

    Notes
    -----
    - Step 1: Direct match checks if the incoming col is in any known synonyms list.
    - Step 2: Partial/fuzzy uses difflib to find the best match in known synonyms.
    - Step 3: GPT fallback tries to forcibly pick exactly one standard name.
      We make it "strict": if GPT returns "None", we do a second attempt to ask
      GPT to pick the "closest" standard. If it still fails, we set it to None.
    - If do_second_pass is True, we call `review_mappings_with_gpt()` to see
      if GPT wants to re-map any columns or fix any Nones.
    """
    logger.debug("Starting match_columns with incoming_cols=%s", incoming_cols)

    ########################################################################
    # STEP 1) DIRECT MATCH
    ########################################################################
    matched_map = {}
    unmatched_after_direct = []

    for col in incoming_cols:
        direct_found = False
        for standard_name, synonyms in known_patterns.items():
            if col in synonyms:
                matched_map[col] = standard_name
                direct_found = True
                logger.debug("Direct match: '%s' -> '%s'", col, standard_name)
                break
        if not direct_found:
            unmatched_after_direct.append(col)

    ########################################################################
    # STEP 2) PARTIAL / FUZZY MATCH
    ########################################################################
    unmatched_after_partial = []
    all_synonyms = []
    for syn_list in known_patterns.values():
        all_synonyms.extend(syn_list)

    for col in unmatched_after_direct:
        potential = difflib.get_close_matches(
            col, all_synonyms, n=1, cutoff=partial_match_threshold
        )
        if potential:
            best_guess = potential[0]
            found_std = None
            for std_name, synonyms in known_patterns.items():
                if best_guess in synonyms:
                    found_std = std_name
                    break
            if found_std:
                matched_map[col] = found_std
                logger.debug("Partial match: '%s' -> '%s' (via '%s')", col, found_std, best_guess)
            else:
                unmatched_after_partial.append(col)
        else:
            unmatched_after_partial.append(col)

    ########################################################################
    # STEP 3) GPT FALLBACK
    ########################################################################
    still_unmatched = []
    for col in unmatched_after_partial:
        gpt_result = _guess_with_gpt(col, known_patterns)
        if gpt_result is not None:
            matched_map[col] = gpt_result
            logger.debug("GPT fallback match: '%s' -> '%s'", col, gpt_result)
        else:
            matched_map[col] = None
            still_unmatched.append(col)
            logger.warning("No GPT match for '%s'; assigned None.", col)

    ########################################################################
    # Optional: Final Review (Second-Pass)
    ########################################################################
    if do_second_pass:
        # If desired, run a final "voting" pass: GPT sees the entire mapping
        # and can optionally correct or fill in any None values.
        matched_map = review_mappings_with_gpt(matched_map, known_patterns)

    return matched_map


def _guess_with_gpt(col, known_patterns):
    """
    Uses GPT to select exactly one standardized name or return None
    if GPT decides there's no valid match or fails to comply.

    We also make it a bit more "strict" by doing a second request
    if GPT tries to return "None".
    """
    possible_standards = list(known_patterns.keys())

    # First attempt: ask GPT for a single match or "None"
    chosen_value = _call_gpt_for_column(col, possible_standards)
    if chosen_value == "None":
        # Second attempt: we try telling GPT it must pick exactly one:
        logger.debug("GPT initially returned 'None' for '%s'; forcing a second attempt.", col)
        chosen_value = _call_gpt_for_column(col, possible_standards, force_pick=True)

    # If GPT still says "None" or fails, we return None
    if chosen_value is None or chosen_value == "None":
        return None

    # If GPT's choice isn't in possible_standards, also return None
    if chosen_value not in possible_standards:
        return None

    return chosen_value


def _call_gpt_for_column(col, possible_standards, force_pick=False):
    """
    Helper that calls GPT one time for a single column, either:
      - "Pick exactly one from the list" if force_pick=True
      - "Return 'None' if no valid match" otherwise
    Returns the GPT's "match" field or None on failure.
    """
    if force_pick:
        # Force GPT to pick one of the standards
        fallback_text = "You must pick exactly one from the list. Do not return None."
    else:
        fallback_text = "Return 'None' if there's no valid match."

    # Build prompt
    prompt_text = f"""
We have a single column: "{col}"
Choose one name from this list:
{possible_standards}
{fallback_text}

JSON format only:
{{
  "match": "<standard_name_or_None>"
}}
"""
    logger.debug("GPT fallback prompt for '%s' (force_pick=%s): %s",
                 col, force_pick, prompt_text.strip())

    chosen_model = choose_model_for_task("short_summarization")
    messages = [
        {"role": "system", "content": "You map column names to standardized names."},
        {"role": "user", "content": prompt_text}
    ]

    try:
        response = call_openai_smart(
            messages=messages,
            model=chosen_model,
            temperature=0.0,
            max_tokens=200,
            max_retries=3
        )
        raw_content = response["choices"][0]["message"]["content"].strip()
        logger.debug("GPT raw response for '%s': %s", col, raw_content)

        parsed_data = _parse_gpt_json(raw_content)
        if "match" in parsed_data:
            return parsed_data["match"]
        return None
    except Exception as e:
        logger.exception("Error calling GPT for column '%s': %s", col, e)
        return None


def review_mappings_with_gpt(matched_map, known_patterns):
    """
    A second-pass or "voting" approach. We feed GPT the entire set of
    {column -> mapped_standard_or_None}, then ask GPT if any should change.

    For any that GPT says to re-map, we accept its suggestion if valid.

    Returns
    -------
    dict
        Possibly updated matched_map after GPT's second-pass review.
    """
    # Build a list of columns + their assigned standard (or None).
    # We'll let GPT correct or fill in any None values if it wants.
    map_items = []
    for col, mapped_std in matched_map.items():
        map_items.append({"column": col, "mapped_to": mapped_std if mapped_std else "None"})

    possible_standards = list(known_patterns.keys())

    review_prompt = f"""
We performed an initial mapping of columns to standard names.
Here is the JSON array of (column, mapped_to) pairs:
{json.dumps(map_items, indent=2)}

You may change any "mapped_to": "None" to a valid standard name from this list:
{possible_standards}

Or fix any that might be incorrect. Respond ONLY in valid JSON array
with objects like: {{"column": "...", "mapped_to": "..."}}, preserving the original 'column' keys.
"""
    logger.debug("Second-pass voting prompt: %s", review_prompt.strip())

    chosen_model = choose_model_for_task("short_summarization")
    messages = [
        {"role": "system", "content": "You can revise column-to-standard mappings."},
        {"role": "user", "content": review_prompt}
    ]

    try:
        response = call_openai_smart(
            messages=messages,
            model=chosen_model,
            temperature=0.0,
            max_tokens=500,
            max_retries=3
        )
        raw_content = response["choices"][0]["message"]["content"].strip()
        logger.debug("GPT second-pass raw response: %s", raw_content)

        revised_array = _parse_gpt_json_array(raw_content)
        if revised_array is None:
            logger.warning("Second-pass GPT review returned invalid data; ignoring.")
            return matched_map

        # Build a new map from GPT's second-pass
        for item in revised_array:
            col_name = item.get("column")
            chosen_std = item.get("mapped_to")
            if not col_name:
                continue
            # If chosen_std is in the known standards, accept it
            if chosen_std in possible_standards:
                matched_map[col_name] = chosen_std
            elif chosen_std == "None":
                matched_map[col_name] = None
        return matched_map
    except Exception as e:
        logger.exception("Error during second-pass GPT review: %s", e)
        return matched_map


def _parse_gpt_json(raw_text):
    """
    Safely parse GPT's JSON response. Returns {} if parsing fails.
    """
    try:
        text = raw_text.strip().strip("```").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        json_str = text[start : end + 1]
        return json.loads(json_str)
    except Exception as e:
        logger.warning("Failed to parse GPT JSON: %s", e)
        return {}


def _parse_gpt_json_array(raw_text):
    """
    Safely parse GPT's JSON response if it returns an ARRAY of objects.
    Returns None if parsing fails.

    Expects something like:
    [
      {"column": "Foo", "mapped_to": "Revenue"},
      {"column": "Bar", "mapped_to": "None"}
    ]
    """
    try:
        text = raw_text.strip().strip("```").strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None
        json_str = text[start : end + 1]
        return json.loads(json_str)
    except Exception as e:
        logger.warning("Failed to parse GPT JSON array: %s", e)
        return None
