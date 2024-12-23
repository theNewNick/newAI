import logging
import numpy as np

logger = logging.getLogger(__name__)

def safe_divide(numerator, denominator):
    try:
        result = numerator / denominator if denominator != 0 else None
        logger.debug(f'Safe divide {numerator} / {denominator} = {result}')
        return result
    except Exception as e:
        logger.error(f'Error in safe_divide: {str(e)}')
        return None

def dcf_analysis(projected_free_cash_flows, wacc, terminal_value, projection_years):
    logger.debug('Starting DCF analysis.')
    discounted_free_cash_flows = [
        fcf / (1 + wacc) ** i for i, fcf in enumerate(projected_free_cash_flows, 1)
    ]
    discounted_terminal_value = terminal_value / (1 + wacc) ** projection_years
    dcf_value = sum(discounted_free_cash_flows) + discounted_terminal_value
    logger.debug('Completed DCF analysis.')
    return dcf_value

def calculate_ratios(financials, benchmarks):
    logger.debug('Starting ratio analysis.')
    current_ratio = safe_divide(financials['current_assets'], financials['current_liabilities'])
    debt_to_equity = safe_divide(financials['total_debt'], financials['shareholders_equity'])
    pe_ratio = safe_divide(financials['market_price'], financials['eps'])
    pb_ratio = safe_divide(financials['market_price'], financials['book_value_per_share'])

    scores = {}
    if debt_to_equity is not None:
        scores['debt_to_equity'] = 1 if debt_to_equity < benchmarks['debt_to_equity'] else -1
    else:
        scores['debt_to_equity'] = 0

    if current_ratio is not None:
        scores['current_ratio'] = 1 if current_ratio > benchmarks['current_ratio'] else -1
    else:
        scores['current_ratio'] = 0

    if pe_ratio is not None:
        scores['pe_ratio'] = 1 if pe_ratio < benchmarks['pe_ratio'] else -1
    else:
        scores['pe_ratio'] = 0

    if pb_ratio is not None:
        scores['pb_ratio'] = 1 if pb_ratio < benchmarks['pb_ratio'] else -1
    else:
        scores['pb_ratio'] = 0

    total_score = sum(scores.values())
    logger.debug(f'Ratio analysis scores: {scores}, Total Score: {total_score}')

    if total_score >= 2:
        normalized_factor2_score = 1
    elif total_score <= -2:
        normalized_factor2_score = -1
    else:
        normalized_factor2_score = 0

    logger.debug(f'Normalized Factor 2 Score: {normalized_factor2_score}')

    return {
        'Current Ratio': current_ratio,
        'Debt-to-Equity Ratio': debt_to_equity,
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'Scores': scores,
        'Total Score': total_score,
        'Normalized Factor 2 Score': normalized_factor2_score
    }

def calculate_cagr(beginning_value, ending_value, periods):
    try:
        if beginning_value <= 0 or periods <= 0:
            logger.warning('Invalid beginning value or periods for CAGR calculation.')
            return None
        cagr = (ending_value / beginning_value) ** (1 / periods) - 1
        logger.debug(f'Calculated CAGR: {cagr}')
        return cagr
    except Exception as e:
        logger.error(f'Error calculating CAGR: {str(e)}')
        return None