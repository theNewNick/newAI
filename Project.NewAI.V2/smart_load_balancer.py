# smart_load_balancer.py

import time
import logging
import openai
import asyncio

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
# ACCOUNTS WITH PER-MODEL LIMITS
# EXAMPLE: You can expand monthly_quota usage to be per-model if you prefer.
###############################################################################
# TIER 1 rate limits:
#   - gpt-4:  RPM=500,   TPM=10,000
#   - gpt-3.5-turbo: RPM=3,500, TPM=200,000
#   - text-embedding-ada-002: RPM=3,000, TPM=1,000,000
#
# TIER 2 rate limits:
#   - gpt-4:  RPM=5,000, TPM=40,000
#   - gpt-3.5-turbo: RPM=3,500, TPM=2,000,000
#   - text-embedding-ada-002: RPM=5,000, TPM=1,000,000
#
# We’ll store them in a nested "limits" dict.

OPENAI_ACCOUNTS = [
    {
        "api_key": OPENAI_API_KEY_1,
        "tier": 1,
        "limits": {
            "gpt-4": {
                "max_rpm": 500,
                "max_tpm": 10_000
            },
            "gpt-3.5-turbo": {
                "max_rpm": 3500,
                "max_tpm": 200_000
            },
            "text-embedding-ada-002": {
                "max_rpm": 3000,
                "max_tpm": 1_000_000
            }
        },
        "monthly_quota": 1_000_000,  # total tokens/month (example)
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_2,
        "tier": 1,
        "limits": {
            "gpt-4": {
                "max_rpm": 500,
                "max_tpm": 10_000
            },
            "gpt-3.5-turbo": {
                "max_rpm": 3500,
                "max_tpm": 200_000
            },
            "text-embedding-ada-002": {
                "max_rpm": 3000,
                "max_tpm": 1_000_000
            }
        },
        "monthly_quota": 1_000_000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_3,
        "tier": 1,
        "limits": {
            "gpt-4": {
                "max_rpm": 500,
                "max_tpm": 10_000
            },
            "gpt-3.5-turbo": {
                "max_rpm": 3500,
                "max_tpm": 200_000
            },
            "text-embedding-ada-002": {
                "max_rpm": 3000,
                "max_tpm": 1_000_000
            }
        },
        "monthly_quota": 1_000_000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_4,
        "tier": 1,
        "limits": {
            "gpt-4": {
                "max_rpm": 500,
                "max_tpm": 10_000
            },
            "gpt-3.5-turbo": {
                "max_rpm": 3500,
                "max_tpm": 200_000
            },
            "text-embedding-ada-002": {
                "max_rpm": 3000,
                "max_tpm": 1_000_000
            }
        },
        "monthly_quota": 1_000_000,
        "monthly_used": 0
    },
    {
        "api_key": OPENAI_API_KEY_5,
        "tier": 2,
        "limits": {
            "gpt-4": {
                "max_rpm": 5000,
                "max_tpm": 40_000
            },
            "gpt-3.5-turbo": {
                "max_rpm": 3500,
                "max_tpm": 2_000_000
            },
            "text-embedding-ada-002": {
                "max_rpm": 5000,
                "max_tpm": 1_000_000
            }
        },
        "monthly_quota": 2_000_000,
        "monthly_used": 0
    }
]


###############################################################################
# USAGE DATA PER ACCOUNT * PER MODEL
###############################################################################
# We track requests_this_minute, tokens_this_minute, last_reset_timestamp
# for each model in each account.
# Example usage_data structure:
#   usage_data[account_index]["gpt-4"] = {
#       "requests_this_minute": 0,
#       "tokens_this_minute": 0,
#       "last_reset_timestamp": ...
#   }

usage_data = []

def init_usage_data():
    """Initialize usage data so each account has usage counters for each model."""
    global usage_data
    usage_data = []
    for acct in OPENAI_ACCOUNTS:
        model_dict = {}
        for model_name in acct["limits"].keys():
            model_dict[model_name] = {
                "requests_this_minute": 0,
                "tokens_this_minute": 0,
                "last_reset_timestamp": time.time()
            }
        usage_data.append(model_dict)

# Call once on import
init_usage_data()


def reset_usage_if_necessary(account_index: int, model: str) -> None:
    """
    Resets this account's per-minute usage counters if more than 60s
    have passed since the last reset for the given model.
    """
    now = time.time()
    elapsed = now - usage_data[account_index][model]["last_reset_timestamp"]
    if elapsed >= 60:
        usage_data[account_index][model]["requests_this_minute"] = 0
        usage_data[account_index][model]["tokens_this_minute"] = 0
        usage_data[account_index][model]["last_reset_timestamp"] = now


def pick_account_smart(model: str) -> int:
    """
    Pick which account to use for the specified model, based on:
      - per-minute capacity (requests + tokens for that model)
      - monthly quota
      - Possibly a tier bonus factor
    Returns the index of the chosen account, or None if none have capacity.
    """
    best_index = None
    best_score = -1

    for i, acct in enumerate(OPENAI_ACCOUNTS):
        # Possibly reset usage for the given model if 60s have passed
        reset_usage_if_necessary(i, model)

        used_rpm = usage_data[i][model]["requests_this_minute"]
        used_tpm = usage_data[i][model]["tokens_this_minute"]
        max_rpm = acct["limits"][model]["max_rpm"]
        max_tpm = acct["limits"][model]["max_tpm"]

        # If we've already hit or exceeded either limit, skip
        if used_rpm >= max_rpm or used_tpm >= max_tpm:
            continue

        monthly_used = acct["monthly_used"]
        monthly_left = acct["monthly_quota"] - monthly_used
        if monthly_left <= 0:
            # Out of monthly tokens
            continue

        monthly_left_ratio = monthly_left / float(acct["monthly_quota"])

        # remaining per-minute ratio for requests & tokens
        remaining_rpm = max_rpm - used_rpm
        remaining_tpm = max_tpm - used_tpm
        rpm_ratio = remaining_rpm / float(max_rpm)
        tpm_ratio = remaining_tpm / float(max_tpm)

        # Example: Tier 2 gets a +0.3 bonus
        tier_bonus = 0.3 if acct["tier"] == 2 else 0.0

        # Simple scoring approach
        score = rpm_ratio + tpm_ratio + monthly_left_ratio + tier_bonus

        if score > best_score:
            best_score = score
            best_index = i

    if best_index is None:
        logger.warning(
            f"[SmartLB] No suitable account found for model='{model}' "
            f"(all at capacity or out of monthly quota)."
        )

    return best_index


###############################################################################
# MAIN FUNCTION: CALL OPENAI (CHAT COMPLETIONS) WITH SMART LOAD BALANCER
###############################################################################
def call_openai_smart(
    messages,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 5
):
    for attempt in range(max_retries):
        account_index = pick_account_smart(model)
        if account_index is None:
            # If no account has capacity, wait and retry
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

            # Update usage for this model
            usage_data[account_index][model]["requests_this_minute"] += 1
            usage_data[account_index][model]["tokens_this_minute"] += used_tokens

            # Update monthly usage (shared bucket in this example)
            acct["monthly_used"] += used_tokens

            logger.debug(
                f"[SmartLB] success: acct_idx={account_index}, tier={acct['tier']}, "
                f"model={model}, used_tokens={used_tokens}, "
                f"req/min={usage_data[account_index][model]['requests_this_minute']}, "
                f"tok/min={usage_data[account_index][model]['tokens_this_minute']}, "
                f"month_used={acct['monthly_used']}"
            )
            return response

        except openai.error.RateLimitError as e:
            # Exponential backoff
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
    for attempt in range(max_retries):
        account_index = pick_account_smart(model)
        if account_index is None:
            logger.warning(f"[Embedding LB] No suitable account found for model={model}; sleeping 5s.")
            time.sleep(5)
            continue

        acct = OPENAI_ACCOUNTS[account_index]
        openai.api_key = acct["api_key"]

        try:
            response = openai.Embedding.create(
                input=input_list,
                model=model
            )

            # Embedding usage often doesn’t include usage info. 
            used_tokens = 0
            if "usage" in response:
                used_tokens = response["usage"].get("total_tokens", 0)

            usage_data[account_index][model]["requests_this_minute"] += 1
            usage_data[account_index][model]["tokens_this_minute"] += used_tokens
            acct["monthly_used"] += used_tokens

            logger.debug(
                f"[Embedding LB] success: acct_idx={account_index}, model={model}, used_tokens={used_tokens}"
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
    for attempt in range(max_retries):
        account_index = pick_account_smart(model)
        if account_index is None:
            logger.debug("[SmartLB Async] No account available; sleeping 5s.")
            await asyncio.sleep(5)
            continue

        acct = OPENAI_ACCOUNTS[account_index]
        openai.api_key = acct["api_key"]

        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            usage_info = response.get("usage", {})
            used_tokens = usage_info.get("total_tokens", 0)

            usage_data[account_index][model]["requests_this_minute"] += 1
            usage_data[account_index][model]["tokens_this_minute"] += used_tokens
            acct["monthly_used"] += used_tokens

            logger.debug(
                f"[SmartLB Async] success: acct_idx={account_index}, tier={acct['tier']}, "
                f"model={model}, used_tokens={used_tokens}"
            )
            return response

        except openai.error.RateLimitError as e:
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
