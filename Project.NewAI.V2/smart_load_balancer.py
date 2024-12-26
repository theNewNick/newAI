# smart_load_balancer.py

import time
import logging
import openai

# We import the 5 API keys from your existing config.py
# so we do NOT store them directly in code.
# This relies on config.py already loading them from .env.
from config import (
    OPENAI_API_KEY_1,
    OPENAI_API_KEY_2,
    OPENAI_API_KEY_3,
    OPENAI_API_KEY_4,
    OPENAI_API_KEY_5,
)

###############################################################################
# LOGGER SETUP
###############################################################################
# You can adjust logging as needed. Currently set to DEBUG for demonstration.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [%(filename)s:%(lineno)d]')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

###############################################################################
# ACCOUNTS & USAGE TRACKING
###############################################################################
# Here we define all 5 accounts. 4 are Tier 1, 1 is Tier 2 (example).
# Adjust "max_rpm" (requests/min), "max_tpm" (tokens/min), "monthly_quota" (tokens/month),
# and "tier" to match your actual plan.
OPENAI_ACCOUNTS = [
    {
        "api_key": OPENAI_API_KEY_1,
        "tier": 1,
        "max_rpm": 500,
        "max_tpm": 10000,
        "monthly_quota": 500000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_2,
        "tier": 1,
        "max_rpm": 500,
        "max_tpm": 10000,
        "monthly_quota": 500000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_3,
        "tier": 1,
        "max_rpm": 500,
        "max_tpm": 10000,
        "monthly_quota": 500000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_4,
        "tier": 1,
        "max_rpm": 500,
        "max_tpm": 10000,
        "monthly_quota": 500000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_5,
        "tier": 2,  # example: the upgraded Tier 2 account
        "max_rpm": 5000,
        "max_tpm": 40000,
        "monthly_quota": 2000000,
        "monthly_used": 0
    }
]

# We'll track usage for each account in-memory for the current minute.
# If you run multiple processes, consider a shared store (e.g., Redis).
usage_data = [
    {
        "requests_this_minute": 0,
        "tokens_this_minute": 0,
        "last_reset_timestamp": time.time()
    }
    for _ in OPENAI_ACCOUNTS
]

###############################################################################
# UTILITY: RESET USAGE IF 60s HAVE ELAPSED
###############################################################################
def reset_usage_if_necessary(account_index: int) -> None:
    """
    Resets this account's per-minute usage counters if more than 60 seconds
    have passed since the last reset.
    """
    now = time.time()
    elapsed = now - usage_data[account_index]["last_reset_timestamp"]
    if elapsed >= 60:
        usage_data[account_index]["requests_this_minute"] = 0
        usage_data[account_index]["tokens_this_minute"] = 0
        usage_data[account_index]["last_reset_timestamp"] = now

###############################################################################
# PICK ACCOUNT BASED ON REMAINING USAGE & MONTHLY QUOTA
###############################################################################
def pick_account_smart() -> int:
    """
    Picks which account to use based on:
      - per-minute capacity (requests + tokens)
      - monthly quota
      - tier (if you want special logic, you can add weighting here).
    Returns the index of the chosen account, or None if none have capacity.
    """
    best_index = None
    best_score = -1

    for i, acct in enumerate(OPENAI_ACCOUNTS):
        # Possibly reset usage for this minute if 60s have passed
        reset_usage_if_necessary(i)

        used_rpm = usage_data[i]["requests_this_minute"]
        used_tpm = usage_data[i]["tokens_this_minute"]

        # If we've already hit or exceeded limits, skip
        if used_rpm >= acct["max_rpm"] or used_tpm >= acct["max_tpm"]:
            continue

        monthly_used = acct["monthly_used"]
        monthly_left = acct["monthly_quota"] - monthly_used
        if monthly_left <= 0:
            # Out of monthly tokens, skip
            continue

        # monthly_left_ratio: how much of monthly quota is left (0..1)
        monthly_left_ratio = monthly_left / float(acct["monthly_quota"])

        # remaining per-minute ratio for requests & tokens
        remaining_rpm = acct["max_rpm"] - used_rpm
        remaining_tpm = acct["max_tpm"] - used_tpm
        rpm_ratio = remaining_rpm / float(acct["max_rpm"])
        tpm_ratio = remaining_tpm / float(acct["max_tpm"])

        # Basic scoring approach: sum these ratios
        # If you want Tier 2 to get a bonus, add e.g. + 0.5 if tier == 2
        tier_bonus = 0.0
        # Example: tier 2 gets an extra 0.3 in the score
        if acct["tier"] == 2:
            tier_bonus = 0.3

        score = rpm_ratio + tpm_ratio + monthly_left_ratio + tier_bonus

        if score > best_score:
            best_score = score
            best_index = i

    if best_index is None:
        logger.warning("No suitable account found (all at capacity or out of monthly quota).")

    return best_index

###############################################################################
# MAIN FUNCTION: CALL OPENAI WITH SMART LOAD BALANCER
###############################################################################
def call_openai_smart(
    messages,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 5
):
    """
    Attempts to call the OpenAI ChatCompletion endpoint using whichever account
    pick_account_smart() chooses. Tracks actual tokens used from the response,
    does exponential backoff on rate-limit errors, and tries up to max_retries.
    """
    for attempt in range(max_retries):
        account_index = pick_account_smart()
        if account_index is None:
            # If no account has capacity at all, wait a bit and retry
            logger.debug("[SmartLB] No account available; sleeping 5s before next attempt.")
            time.sleep(5)
            continue

        acct = OPENAI_ACCOUNTS[account_index]
        openai.api_key = acct["api_key"]

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            usage_info = response.get("usage", {})
            used_tokens = usage_info.get("total_tokens", 0)

            # Update per-minute counters
            usage_data[account_index]["requests_this_minute"] += 1
            usage_data[account_index]["tokens_this_minute"] += used_tokens

            # Update monthly usage
            acct["monthly_used"] += used_tokens

            logger.debug(
                f"[SmartLB] Success with account_idx={account_index}, tier={acct['tier']}, "
                f"used_tokens={used_tokens}, requests_this_minute={usage_data[account_index]['requests_this_minute']}, "
                f"tokens_this_minute={usage_data[account_index]['tokens_this_minute']}, "
                f"monthly_used={acct['monthly_used']}, model={model}"
            )
            return response

        except openai.error.RateLimitError as e:
            # Exponential backoff on rate-limit
            backoff_seconds = min(60, 2 ** attempt)
            logger.warning(
                f"[SmartLB] RateLimitError on account_idx={account_index}, attempt={attempt}: {e}. "
                f"Sleeping {backoff_seconds}s."
            )
            time.sleep(backoff_seconds)
            continue

        except openai.error.OpenAIError as e:
            logger.error(
                f"[SmartLB] OpenAIError on account_idx={account_index}, attempt={attempt}: {e}. "
                "Will retry after short sleep."
            )
            time.sleep(3)
            continue

    # If all retries fail, raise an Exception
    raise Exception("[SmartLB] All retries exhausted in call_openai_smart().")
