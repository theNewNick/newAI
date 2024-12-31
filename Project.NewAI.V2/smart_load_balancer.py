import time
import logging
import openai
import asyncio  # Added for the async call

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

        # If we've already hit or exceeded either limit, skip
        if used_rpm >= acct["max_rpm"] or used_tpm >= acct["max_tpm"]:
            continue

        monthly_used = acct["monthly_used"]
        monthly_left = acct["monthly_quota"] - monthly_used
        if monthly_left <= 0:
            # Out of monthly tokens
            continue

        # monthly_left_ratio: how much of monthly quota is left (0..1)
        monthly_left_ratio = monthly_left / float(acct["monthly_quota"])

        # remaining per-minute ratio for requests & tokens
        remaining_rpm = acct["max_rpm"] - used_rpm
        remaining_tpm = acct["max_tpm"] - used_tpm
        rpm_ratio = remaining_rpm / float(acct["max_rpm"])
        tpm_ratio = remaining_tpm / float(acct["max_tpm"])

        # Example tier bonus: Tier 2 gets +0.3
        tier_bonus = 0.3 if acct["tier"] == 2 else 0.0

        # Simple scoring approach
        score = rpm_ratio + tpm_ratio + monthly_left_ratio + tier_bonus

        if score > best_score:
            best_score = score
            best_index = i

    if best_index is None:
        logger.warning("No suitable account found (all at capacity or out of monthly quota).")

    return best_index

###############################################################################
# MAIN FUNCTION: CALL OPENAI (CHAT) WITH SMART LOAD BALANCER
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
            logger.debug("[SmartLB] No account available; sleeping 5s.")
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
                f"[SmartLB] Success with acct_idx={account_index}, tier={acct['tier']}, "
                f"used_tokens={used_tokens}, req/min={usage_data[account_index]['requests_this_minute']}, "
                f"tok/min={usage_data[account_index]['tokens_this_minute']}, "
                f"month_used={acct['monthly_used']}, model={model}"
            )
            return response

        except openai.error.RateLimitError as e:
            # Exponential backoff on rate-limit
            backoff = min(60, 2 ** attempt)
            logger.warning(
                f"[SmartLB] RateLimitError on acct_idx={account_index}, attempt={attempt}: {e}. "
                f"Sleeping {backoff}s."
            )
            time.sleep(backoff)
            continue

        except openai.error.OpenAIError as e:
            logger.error(
                f"[SmartLB] OpenAIError on acct_idx={account_index}, attempt={attempt}: {e}. "
                "Retrying in 3s."
            )
            time.sleep(3)
            continue

    raise Exception("[SmartLB] All retries exhausted in call_openai_smart().")

###############################################################################
# NEW FUNCTION: CALL OPENAI (EMBEDDING) WITH SMART LOAD BALANCER
###############################################################################
def call_openai_embedding_smart(
    input_list,
    model: str = "text-embedding-ada-002",
    max_retries: int = 5
):
    """
    Similar to call_openai_smart, but uses openai.Embedding.create
    instead of openai.ChatCompletion.create.
    Tracks usage and does exponential backoff on rate-limit errors.
    """
    for attempt in range(max_retries):
        account_index = pick_account_smart()
        if account_index is None:
            logger.warning("[Embedding LB] No suitable account found; sleeping 5s.")
            time.sleep(5)
            continue

        acct = OPENAI_ACCOUNTS[account_index]
        openai.api_key = acct["api_key"]

        try:
            response = openai.Embedding.create(
                input=input_list,
                model=model
            )

            # Embedding responses often do not include usage info.
            # You may set used_tokens=0 or attempt to parse from response if available.
            used_tokens = 0

            usage_data[account_index]["requests_this_minute"] += 1
            usage_data[account_index]["tokens_this_minute"] += used_tokens
            acct["monthly_used"] += used_tokens

            logger.debug(
                f"[Embedding LB] Success with acct_idx={account_index}, model={model}"
            )
            return response

        except openai.error.RateLimitError as e:
            backoff = min(60, 2 ** attempt)
            logger.warning(
                f"[Embedding LB] RateLimitError on acct_idx={account_index}, attempt={attempt}: {e}. "
                f"Sleeping {backoff}s."
            )
            time.sleep(backoff)
            continue

        except openai.error.OpenAIError as e:
            logger.error(
                f"[Embedding LB] OpenAIError on acct_idx={account_index}, attempt={attempt}: {e}. "
                "Retrying in 3s."
            )
            time.sleep(3)
            continue

    raise Exception("[Embedding LB] All retries exhausted in call_openai_embedding_smart.")

###############################################################################
# ASYNCHRONOUS: CALL OPENAI (CHAT) WITH SMART LOAD BALANCER
###############################################################################
async def call_openai_smart_async(
    messages,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 5
):
    """
    Asynchronous version of call_openai_smart, using Python's `asyncio` and
    openai's async endpoints, for concurrency. 
    Make sure you have a recent version of `openai` that supports .acreate().

    Tracks usage, handles rate-limit backoff with await, and tries up to max_retries.
    """
    for attempt in range(max_retries):
        account_index = pick_account_smart()
        if account_index is None:
            logger.debug("[SmartLB Async] No account available; sleeping 5s.")
            await asyncio.sleep(5)
            continue

        acct = OPENAI_ACCOUNTS[account_index]
        openai.api_key = acct["api_key"]

        try:
            # The "acreate()" method is the async version of openai.ChatCompletion.create
            response = await openai.ChatCompletion.acreate(
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
                f"[SmartLB Async] Success with acct_idx={account_index}, tier={acct['tier']}, "
                f"used_tokens={used_tokens}, model={model}"
            )
            return response

        except openai.error.RateLimitError as e:
            # Exponential backoff on rate-limit, but using async sleeps
            backoff = min(60, 2 ** attempt)
            logger.warning(
                f"[SmartLB Async] RateLimitError on acct_idx={account_index}, attempt={attempt}: {e}. "
                f"Sleeping {backoff}s."
            )
            await asyncio.sleep(backoff)
            continue

        except openai.error.OpenAIError as e:
            logger.error(
                f"[SmartLB Async] OpenAIError on acct_idx={account_index}, attempt={attempt}: {e}. "
                "Retrying in 3s."
            )
            await asyncio.sleep(3)
            continue

    raise Exception("[SmartLB Async] All retries exhausted in call_openai_smart_async().")
