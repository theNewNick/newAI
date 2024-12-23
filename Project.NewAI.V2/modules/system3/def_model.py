import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DCFModel:
    def __init__(self, initial_values, assumptions):
        self.initial_values = initial_values or {}
        self.assumptions = assumptions or {}
        self.years = 5
        self.fcf_values = []
        self.discounted_fcf = []
        self.intrinsic_value = 0
        self.intrinsic_value_per_share = 0
        self.discounted_terminal_value = 0
        self.terminal_value = 0
        self.projections = pd.DataFrame({'Year': range(1, self.years + 1)})

    def project_fcf(self):
        logger.debug("Projecting FCF")
        revenue = self.initial_values.get('Revenue', 0)
        revenue_growth_rate = self.assumptions.get('revenue_growth_rate', 0.05)
        tax_rate = self.assumptions.get('tax_rate', 0.21)
        cogs_pct = self.assumptions.get('cogs_pct', 0.6)
        operating_expenses_pct = self.assumptions.get('operating_expenses_pct', 0.2)
        depreciation_pct = self.assumptions.get('depreciation_pct', 0.05)
        capex_pct = self.assumptions.get('capex_pct', 0.05)
        nwc_pct = self.assumptions.get('nwc_pct', 0.2)

        prev_nwc = revenue * nwc_pct

        for year in range(self.years):
            projected_revenue = revenue * ((1 + revenue_growth_rate) ** (year + 1))
            cogs = projected_revenue * cogs_pct
            gross_profit = projected_revenue - cogs
            operating_expenses = projected_revenue * operating_expenses_pct
            depreciation = projected_revenue * depreciation_pct
            ebit = gross_profit - operating_expenses - depreciation
            nopat = ebit * (1 - tax_rate)
            capex = projected_revenue * capex_pct
            nwc = projected_revenue * nwc_pct
            change_in_nwc = nwc - prev_nwc
            prev_nwc = nwc
            fcf = nopat + depreciation - capex - change_in_nwc
            self.fcf_values.append(fcf)
            self.projections.loc[year, 'Revenue'] = projected_revenue
            self.projections.loc[year, 'COGS'] = cogs
            self.projections.loc[year, 'Operating Expenses'] = operating_expenses
            self.projections.loc[year, 'Depreciation'] = depreciation
            self.projections.loc[year, 'EBIT'] = ebit
            self.projections.loc[year, 'NOPAT'] = nopat
            self.projections.loc[year, 'CAPEX'] = capex
            self.projections.loc[year, 'Change in NWC'] = change_in_nwc
            self.projections.loc[year, 'FCF'] = fcf
            logger.debug(f"Year {year + 1}: Projected Revenue = {projected_revenue}, FCF = {fcf}")

    def calculate_terminal_value(self):
        terminal_growth_rate = self.assumptions.get('terminal_growth_rate', 0.02)
        wacc = self.assumptions.get('wacc', 0.10)
        last_fcf = self.fcf_values[-1]
        logger.debug(f"Calculating terminal value with wacc={wacc}, terminal_growth_rate={terminal_growth_rate}, last_fcf={last_fcf}")
        self.terminal_value = last_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    def discount_cash_flows(self):
        logger.debug("Discounting cash flows")
        wacc = self.assumptions.get('wacc', 0.10)
        for i, fcf in enumerate(self.fcf_values):
            discounted = fcf / ((1 + wacc) ** (i + 1))
            self.discounted_fcf.append(discounted)
        self.discounted_terminal_value = self.terminal_value / ((1 + wacc) ** self.years)

    def calculate_intrinsic_value(self):
        logger.debug("Calculating intrinsic value")
        self.intrinsic_value = sum(self.discounted_fcf) + self.discounted_terminal_value
        shares_outstanding = self.assumptions.get('shares_outstanding', 1)
        self.intrinsic_value_per_share = self.intrinsic_value / shares_outstanding

    def run_model(self):
        logger.debug("Running DCF model")
        self.project_fcf()
        self.calculate_terminal_value()
        self.discount_cash_flows()
        self.calculate_intrinsic_value()

    def get_results(self):
        logger.debug("Fetching DCF model results")
        return {
            "intrinsic_value": self.intrinsic_value,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "discounted_fcf": self.discounted_fcf,
            "discounted_terminal_value": self.discounted_terminal_value,
            "fcf_values": self.fcf_values,
            "projections": self.projections.to_dict(orient='records')
        }