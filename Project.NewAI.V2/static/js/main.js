// main.js
// ---------------------------------------------------------------------------
// This file handles all AJAX requests to the backend and updates the dashboard
// with real-time data. It also includes chatbot and scenario analysis features.
//
// It implements the multi-state approach for each tile:
//   - Default => .state-default
//   - Collapsed => .state-collapsed
//   - Enlarged => .state-enlarged
//
// Additionally, it uses a 3-tier grid layout (top, middle, bottom rows)
// for the expanded tile. The tile in the middle row is .expanded + .enlarged,
// while the others become minimized + .collapsed.
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  // 1. Load data into each tile
  initDashboard();

  // 2. Load the agentic base case from system3
  initBaseCase();

  // 2. Chatbot functionality
  initChatbot();

  // 3. Scenario Analysis functionality
  initScenarioAnalysis();

  // 4. Three-row tile layout logic
  initThreeRowTileLayout();
});

// ---------------------------------------------------------------------------
// 1. Data Fetching + Populating Tiles
// ---------------------------------------------------------------------------
function initDashboard() {
  // (A) Company Report
  fetch('/system1/company_report_data')
    .then(res => res.json())
    .then(data => {
      const defaultKey = document.querySelector('#company-report-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#company-report-container .state-collapsed .tiny-data');

      // --- NEW LOGIC FOR pct_change_12mo ---
      const stockPrice = (data.stock_price !== undefined && data.stock_price !== null)
        ? data.stock_price 
        : 145.32; // fallback if none
      const pctChange = (data.pct_change_12mo !== undefined && data.pct_change_12mo !== null)
        ? data.pct_change_12mo
        : null;

      let displayText = '';
      if (stockPrice === null) {
        // Fallback to a default string if we truly have no data
        displayText = '$145.32';
      } else {
        // Format the price to 2 decimals (if it's a number)
        const priceStr = (typeof stockPrice === 'number')
          ? stockPrice.toFixed(2)
          : stockPrice;

        if (pctChange !== null) {
          const arrow = (pctChange >= 0) ? '▲' : '▼';
          const sign = (pctChange >= 0) ? '+' : '';
          // CHANGED: Decide which color class to use
          let colorClass = (pctChange >= 0) ? 'positive-change' : 'negative-change';

          // Build an HTML string with <span> for color
          displayText = `<span class="${colorClass}">${priceStr} ${arrow}${sign}${pctChange.toFixed(1)}%</span>`;
        } else {
          // If we have a price but no percent change
          displayText = priceStr;
        }
      }

      // NEW: set innerHTML to display the <span>
      if (defaultKey) defaultKey.innerHTML = displayText;
      if (collapsedKey) collapsedKey.innerHTML = displayText;

      const fullDiv = document.getElementById('company-report-full');
      if (fullDiv) {
        fullDiv.innerHTML = `
          <p><strong>Executive Summary:</strong> ${data.executive_summary}</p>
          <p><strong>Company Summary:</strong> ${data.company_summary}</p>
          <p><strong>Industry Summary:</strong> ${data.industry_summary}</p>
          <p><strong>Risk Considerations:</strong> ${data.risk_considerations}</p>
        `;
      }
    })
    .catch(err => {
      console.error("Error fetching company report data:", err);
    });

  // (B) Financial Analysis
  fetch('/system1/financial_analysis_data')
    .then(res => res.json())
    .then(data => {
      // 1) Default & Collapsed State: DCF as the key metric
      const dcfValueDefault = document.querySelector('#financial-analysis-container .state-default .key-metric');
      const dcfValueCollapsed = document.querySelector('#financial-analysis-container .state-collapsed .tiny-data');
      const dcfNumber = data.dcf_intrinsic_value ? parseFloat(data.dcf_intrinsic_value) : 127.50;
      const dcfStr = `$${dcfNumber.toFixed(2)}`;

      if (dcfValueDefault) dcfValueDefault.textContent = dcfStr;
      if (dcfValueCollapsed) dcfValueCollapsed.textContent = dcfStr;

      // 2) Default State: One-liner is set in HTML, but we also have a bullet chart
      // We'll plot annual_profit_margins vs. industry_profit_margin
      const profitMarginCanvas = document.getElementById('profitMarginBulletChart');
      if (profitMarginCanvas && data.annual_profit_margins && data.industry_profit_margin) {
        const ctx = profitMarginCanvas.getContext('2d');

        const marginLabels = Object.keys(data.annual_profit_margins);
        const marginData = marginLabels.map(yr => (data.annual_profit_margins[yr] * 100));
        const industryTarget = data.industry_profit_margin * 100;

        // Build a bar + line overlay
        new Chart(ctx, {
          type: 'bar',
          data: {
            labels: marginLabels,
            datasets: [
              {
                label: 'Profit Margin (%)',
                data: marginData,
                backgroundColor: 'rgba(0,123,255,0.5)'
              },
              {
                label: 'Industry Standard',
                data: marginLabels.map(() => industryTarget),
                type: 'line',
                borderColor: 'red',
                borderWidth: 2,
                fill: false
              }
            ]
          },
          options: {
            responsive: false,
            maintainAspectRatio: false,
            scales: {
              y: {
                beginAtZero: true,
                max: 50 // adjust if you expect margins can go > 50%
              }
            },
            plugins: {
              legend: {
                display: false // or true if you want to show "Profit Margin" + "Industry Standard"
              }
            }
          }
        });
      }

      // 3) Expanded State
      // We'll create a more detailed HTML for #financial-analysis-full
      const fullDiv = document.getElementById('financial-analysis-full');
      if (fullDiv) {
        // DCF
        let expandedHTML = `<h3>DCF Intrinsic Value: <strong>${dcfStr}</strong></h3>`;

        // Ratios vs. Benchmarks (like a table)
        expandedHTML += `<h4>Ratios vs. Benchmarks</h4>`;
        if (data.ratios && data.industry_benchmarks) {
          expandedHTML += `<table class="table table-sm">
            <thead>
              <tr>
                <th>Ratio</th>
                <th>Company</th>
                <th>Industry</th>
              </tr>
            </thead>
            <tbody>`;

          // We'll assume data.ratios is an object with e.g. "profit_margin", "current_ratio_calc", ...
          // We'll match them to data.industry_benchmarks by the same key if it exists
          for (let ratioName in data.ratios) {
            // If ratio is numeric
            const companyVal = data.ratios[ratioName];
            const industryVal = (data.industry_benchmarks[ratioName] !== undefined)
              ? data.industry_benchmarks[ratioName]
              : null;

            // Decide if it's a fraction that needs *100 or not
            // For demonstration, let's do that if ratioName includes 'margin'
            let displayCompany = companyVal;
            let displayIndustry = industryVal;
            if (ratioName.toLowerCase().includes('margin')) {
              displayCompany = (companyVal * 100).toFixed(2) + '%';
              if (industryVal !== null) displayIndustry = (industryVal * 100).toFixed(2) + '%';
            } else if (typeof companyVal === 'number') {
              displayCompany = companyVal.toFixed(3);
              if (industryVal !== null) displayIndustry = industryVal.toFixed(3);
            }

            expandedHTML += `
              <tr>
                <td>${ratioName}</td>
                <td>${(displayCompany !== null && displayCompany !== undefined) ? displayCompany : 'N/A'}</td>
                <td>${(displayIndustry !== null && displayIndustry !== undefined) ? displayIndustry : 'N/A'}</td>
              </tr>
            `;
          }

          expandedHTML += `</tbody></table>`;
        } else {
          expandedHTML += `<p>(No ratio data available)</p>`;
        }

        // For demonstration, let's also build bullet charts for each ratio individually
        // We'll create a container for them
        expandedHTML += `<div id="ratio-bullet-charts" style="display:flex; flex-wrap:wrap; gap:20px;"></div>`;

        // Finally, the Time Series (quarterly) if available
        expandedHTML += `<h4>Last 4 Quarters</h4>`;
        if (data.time_series_analysis && data.time_series_analysis.quarterly) {
          // We'll create a canvas
          expandedHTML += `<canvas id="quarterlyMarginChart" width="400" height="200"></canvas>`;
        }

        fullDiv.innerHTML = expandedHTML;

        // Now we do the bullet charts for each ratio
        if (data.ratios && data.industry_benchmarks) {
          const ratioContainer = document.getElementById('ratio-bullet-charts');
          for (let ratioName in data.ratios) {
            // Create a small <div> with a canvas
            const div = document.createElement('div');
            div.style.width = '200px';
            div.style.height = '120px';
            div.style.border = '1px solid #ccc';
            div.style.padding = '5px';
            div.style.boxSizing = 'border-box';

            const c = document.createElement('canvas');
            c.width = 200;
            c.height = 80;
            div.appendChild(c);
            ratioContainer.appendChild(div);

            // We'll build a small bar + line overlay chart
            const companyVal = data.ratios[ratioName];
            const industryVal = (data.industry_benchmarks[ratioName] !== undefined)
              ? data.industry_benchmarks[ratioName]
              : 0;

            let labelVal = ratioName;
            let companyDisplay = companyVal;
            let industryDisplay = industryVal;

            // If it's a margin, multiply by 100
            if (ratioName.toLowerCase().includes('margin')) {
              companyDisplay = companyVal * 100;
              industryDisplay = industryVal * 100;
            }

            new Chart(c.getContext('2d'), {
              type: 'bar',
              data: {
                labels: [labelVal],
                datasets: [
                  {
                    label: labelVal,
                    data: [companyDisplay],
                    backgroundColor: 'rgba(0,123,255,0.5)'
                  },
                  {
                    label: 'Benchmark',
                    data: [industryDisplay],
                    type: 'line',
                    borderColor: 'red',
                    borderWidth: 2,
                    fill: false
                  }
                ]
              },
              options: {
                responsive: false,
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true
                  }
                },
                plugins: {
                  legend: {
                    display: false
                  }
                }
              }
            });
          }
        }

        // Next, if we have last-4-quarters data, render a small line chart
        if (data.time_series_analysis && data.time_series_analysis.quarterly) {
          const quarterlyCanvas = document.getElementById('quarterlyMarginChart');
          if (quarterlyCanvas) {
            const quarters = data.time_series_analysis.quarterly.map(d => d.quarter);
            // For demonstration, let's chart the profit_margin as a percentage
            const pmValues = data.time_series_analysis.quarterly.map(d => (d.profit_margin * 100));

            new Chart(quarterlyCanvas.getContext('2d'), {
              type: 'line',
              data: {
                labels: quarters,
                datasets: [{
                  label: 'Profit Margin (%)',
                  data: pmValues,
                  borderColor: 'blue',
                  fill: false
                }]
              },
              options: {
                responsive: false,
                maintainAspectRatio: false,
                scales: {
                  y: { beginAtZero: true }
                }
              }
            });
          }
        }
      }
    })
    .catch(err => {
      console.error("Error fetching financial analysis data:", err);
    });

  // (C) Sentiment Analysis
  fetch('/system1/sentiment_data')
    .then(res => res.json())
    .then(data => {
      // --- New Collapsed State logic ---
      const collapsedElem = document.querySelector(
        '#sentiment-analysis-container .state-collapsed .tiny-data'
      );
      if (collapsedElem) {
        const score = (data.composite_score !== undefined)
          ? data.composite_score
          : 0;
        const sign = (score > 0) ? '+' : (score < 0) ? '-' : '';
        const absVal = Math.abs(score).toFixed(2);
        const colorClass = (score > 0)
          ? 'positive-score'
          : (score < 0)
            ? 'negative-score'
            : 'neutral-score';

        collapsedElem.innerHTML = `
          <span class="${colorClass}">
            Composite: ${sign}${absVal}
          </span>
        `;
      }

      // --- New Default State logic (with gradient bar) ---
      const defaultKeyElem = document.querySelector(
        '#sentiment-analysis-container .state-default .key-metric'
      );
      if (defaultKeyElem) {
        const score = (data.composite_score !== undefined)
          ? data.composite_score
          : 0;
        const sign = (score > 0) ? '+' : (score < 0) ? '-' : '';
        const absVal = Math.abs(score).toFixed(2);
        const colorClass = (score > 0)
          ? 'positive-score'
          : (score < 0)
            ? 'negative-score'
            : 'neutral-score';

        defaultKeyElem.innerHTML = `
          <span class="${colorClass}">
            Composite: ${sign}${absVal}
          </span>
        `;
      }

      // Position marker in the default state's gradient bar (#compositeMarker)
      const markerElem = document.getElementById('compositeMarker');
      if (markerElem) {
        const score = (data.composite_score !== undefined)
          ? data.composite_score
          : 0;
        // Map score (-1 to +1) to 0%..100%
        const leftPercent = ((score + 1) / 2) * 100;
        markerElem.style.left = leftPercent + '%';
      }

      // --- New Expanded State logic: sub-sentiments + gradient bars ---
      const fullDiv = document.getElementById('sentiment-analysis-full');
      if (fullDiv) {
        let htmlContent = '';

        // Helper to render each sub-sentiment block
        function renderSubSentiment(title, obj) {
          const s = (obj && obj.score !== undefined) ? obj.score : 0;
          const sign = (s > 0) ? '+' : (s < 0) ? '-' : '';
          const absVal = Math.abs(s).toFixed(2);
          const cclass = (s > 0)
            ? 'positive-score'
            : (s < 0)
              ? 'negative-score'
              : 'neutral-score';
          const explanation = (obj && obj.explanation) ? obj.explanation : '';
          // Create unique bar/marker IDs for each block
          const barId = `bar-${title.replace(/\s+/g, '-')}`;
          const markerId = `marker-${title.replace(/\s+/g, '-')}`;

          return `
            <h4>${title}</h4>
            <p>
              <span class="${cclass}">
                Score: ${sign}${absVal}
              </span>
              – ${explanation}
            </p>
            <div class="sentiment-bar" id="${barId}">
              <div class="sentiment-marker" id="${markerId}"></div>
            </div>
          `;
        }

        // Build sub-sections for earnings, industry, economic
        htmlContent += renderSubSentiment(
          "Earnings Call Sentiment",
          data.earnings_call_sentiment
        );
        htmlContent += renderSubSentiment(
          "Industry Report Sentiment",
          data.industry_report_sentiment
        );
        htmlContent += renderSubSentiment(
          "Economic Report Sentiment",
          data.economic_report_sentiment
        );

        // Inject into the expanded area
        fullDiv.innerHTML = htmlContent;

        // Now, position each sub-sentiment marker
        function setMarker(score, markerId) {
          const leftPct = ((score + 1) / 2) * 100;
          const el = document.getElementById(markerId);
          if (el) el.style.left = leftPct + '%';
        }

        // E.g. earnings, industry, economic
        if (data.earnings_call_sentiment && data.earnings_call_sentiment.score !== undefined) {
          setMarker(data.earnings_call_sentiment.score, 'marker-Earnings-Call-Sentiment');
        }
        if (data.industry_report_sentiment && data.industry_report_sentiment.score !== undefined) {
          setMarker(data.industry_report_sentiment.score, 'marker-Industry-Report-Sentiment');
        }
        if (data.economic_report_sentiment && data.economic_report_sentiment.score !== undefined) {
          setMarker(data.economic_report_sentiment.score, 'marker-Economic-Report-Sentiment');
        }
      }
     })
    .catch(err => {
      console.error("Error fetching sentiment data:", err);
    });

  // (D) Data Visualizations
function initDataVisualizations() {
  fetch('/system1/data_visualizations_data')
    .then(res => res.json())
    .then(data => {
      console.log("Data Viz data:", data);

      // 1) Collapsed snippet => TTM Revenue snippet
      updateDataVizCollapsed(data);

      // 2) Default snippet => TTM snippet + scatter
      updateDataVizDefault(data);

      // 3) Expanded charts => multiple bar/line/stacked
      buildDataVizExpandedCharts(data);
    })
    .catch(err => {
      console.error("Error fetching Data Viz data:", err);
    });
}

function updateDataVizCollapsed(data) {
  // Suppose data includes: ttm_current, ttm_previous
  const ttmCurrent = data.ttm_current;
  const ttmPrevious = data.ttm_previous;
  if (!ttmCurrent || !ttmPrevious) return;

  const pctChange = ((ttmCurrent - ttmPrevious) / ttmPrevious) * 100;
  const arrow = pctChange >= 0 ? '▲' : '▼';
  const sign = pctChange >= 0 ? '+' : '';

  // e.g. round or floor the number
  const formattedTTM = formatNumber(Math.floor(ttmCurrent));
  const snippet = `TTM Revenue: $${formattedTTM} ${arrow}${sign}${pctChange.toFixed(1)}%`;

  const collapsedElem = document.querySelector('#data-visualizations-container .state-collapsed .tiny-data');
  if (collapsedElem) collapsedElem.textContent = snippet;
}

function updateDataVizDefault(data) {
  const ttmCurrent = data.ttm_current;
  const ttmPrevious = data.ttm_previous;
  if (!ttmCurrent || !ttmPrevious) return;

  // Calculate % change
  const pctChange = ((ttmCurrent - ttmPrevious) / ttmPrevious) * 100;
  const arrow = (pctChange >= 0) ? '▲' : '▼';
  const sign = (pctChange >= 0) ? '+' : '';
  // Decide color class
  const colorClass = (pctChange >= 0) ? 'positive-change' : 'negative-change';

  // Format TTM (remove decimals, etc.)
  const formattedTTM = formatNumber(Math.floor(ttmCurrent));

  // Build snippet with colored percentage
  const snippet = `
    TTM: $${formattedTTM}
    <span class="${colorClass}">
      ${arrow}${sign}${pctChange.toFixed(1)}%
    </span>
  `;

  // Insert snippet into the key-metric element using innerHTML
  const keyMetricElem = document.querySelector('#data-visualizations-container .state-default .key-metric');
  if (keyMetricElem) {
    keyMetricElem.innerHTML = snippet;
  }

  // Build a scatter plot for annual_revenue in <canvas id="annualRevenueScatter">
  const scatterCtx = document.getElementById('annualRevenueScatter')?.getContext('2d');
  if (!scatterCtx || !data.annual_revenue) return;

  const scatterData = data.annual_revenue.map(item => ({
    x: item.year,
    y: Math.round(item.value / 1e6) // e.g. in millions
  }));

  new Chart(scatterCtx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Annual Revenue (millions)',
        data: scatterData,
        backgroundColor: 'blue'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Year' }, ticks: { stepSize: 1 } },
        y: { title: { display: true, text: 'Millions' } }
      }
    }
  });
}

function buildDataVizExpandedCharts(data) {
  const yearlyData = data.yearly_data;
  if (!yearlyData) return;
  const years = yearlyData.map(d => d.year);

  // 1) Assets vs. Liabilities (bar)
  const assetsLiabilitiesCtx = document.getElementById('assetsLiabilitiesBar')?.getContext('2d');
  if (assetsLiabilitiesCtx) {
    const assets = yearlyData.map(d => d.assets);
    const liab = yearlyData.map(d => d.liabilities);
    new Chart(assetsLiabilitiesCtx, {
      type: 'bar',
      data: {
        labels: years,
        datasets: [
          { label: 'Assets', data: assets, backgroundColor: 'blue' },
          { label: 'Liabilities', data: liab, backgroundColor: 'red' }
        ]
      },
      options: { responsive: true }
    });
  }

  // 2) Operating Cash Flow (line)
  const ocfCtx = document.getElementById('operatingCashFlowLine')?.getContext('2d');
  if (ocfCtx) {
    const ocf = yearlyData.map(d => d.operating_cash_flow);
    new Chart(ocfCtx, {
      type: 'line',
      data: {
        labels: years,
        datasets: [{
          label: 'Operating CF',
          data: ocf,
          borderColor: 'green',
          fill: false
        }]
      },
      options: { responsive: true }
    });
  }

  // 3) Revenue & Net Income (bar)
  const revNiCtx = document.getElementById('revenueNetIncomeBar')?.getContext('2d');
  if (revNiCtx) {
    const rev = yearlyData.map(d => d.revenue);
    const ni = yearlyData.map(d => d.net_income);
    new Chart(revNiCtx, {
      type: 'bar',
      data: {
        labels: years,
        datasets: [
          { label: 'Revenue', data: rev, backgroundColor: 'blue' },
          { label: 'Net Income', data: ni, backgroundColor: 'orange' }
        ]
      },
      options: { responsive: true }
    });
  }

  // 4) Expenses Breakdown (stacked bar)
  const expCtx = document.getElementById('expensesStackedBar')?.getContext('2d');
  if (expCtx) {
    const cogs = yearlyData.map(d => d.cogs);
    const sga = yearlyData.map(d => d.sga);
    const dep = yearlyData.map(d => d.depreciation);
    const intExp = yearlyData.map(d => d.interest_expense);
    const tax = yearlyData.map(d => d.income_tax);

    new Chart(expCtx, {
      type: 'bar',
      data: {
        labels: years,
        datasets: [
          { label: 'COGS', data: cogs, backgroundColor: '#a1cfff' },
          { label: 'SG&A', data: sga, backgroundColor: '#ffa1c4' },
          { label: 'Depreciation', data: dep, backgroundColor: '#c3ffa1' },
          { label: 'Interest', data: intExp, backgroundColor: '#ffc1a1' },
          { label: 'Income Tax', data: tax, backgroundColor: '#d1a1ff' }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { stacked: true },
          y: { stacked: true }
        }
      }
    });
  }

  // 5) Income Breakdown (stacked bar)
  const incCtx = document.getElementById('incomeStackedBar')?.getContext('2d');
  if (incCtx) {
    const rev = yearlyData.map(d => d.revenue);
    const ni = yearlyData.map(d => d.net_income);
    const gp = yearlyData.map(d => (d.revenue - d.cogs));
    const ebitda = yearlyData.map(d => d.ebitda);
    const ebit = yearlyData.map(d => d.ebit);
    const ebt = yearlyData.map(d => d.ebt);

    new Chart(incCtx, {
      type: 'bar',
      data: {
        labels: years,
        datasets: [
          { label: 'Revenue', data: rev, backgroundColor: '#cce5ff' },
          { label: 'Gross Profit', data: gp, backgroundColor: '#d1ffd6' },
          { label: 'EBITDA', data: ebitda, backgroundColor: '#ffffb1' },
          { label: 'EBIT', data: ebit, backgroundColor: '#ffd9b1' },
          { label: 'EBT', data: ebt, backgroundColor: '#ffcce7' },
          { label: 'Net Income', data: ni, backgroundColor: '#e0ccff' }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { stacked: true },
          y: { stacked: true }
        }
      }
    });
  }
}

function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// (E) Final Recommendation
fetch('/system1/final_recommendation')
  .then(res => res.json())
  .then(finalData => {
    // In expanded mode, we also want the DCF, ratios, and sentiment details.
    // We'll fetch them in parallel, then combine.
    return Promise.all([
      Promise.resolve(finalData),
      fetch('/system1/financial_analysis_data').then(r => r.json()),
      fetch('/system1/sentiment_data').then(r => r.json())
    ]);
  })
  .then(([finalData, finAnalysisData, sentimentData]) => {
    //-----------------------------------------------------------------------
    // 1) Collapsed State
    //    - "#final-collapsed-rec" => "BUY", "HOLD", or "SELL"
    //-----------------------------------------------------------------------
    const collapsedElem = document.querySelector(
      '#final-recommendation-container .state-collapsed .tiny-data'
    );
    if (collapsedElem) {
      collapsedElem.textContent = finalData.recommendation || 'BUY';
    }

    //-----------------------------------------------------------------------
    // 2) Default State
    //    - "#final-score" => "Score: X"
    //    - .one-liner => "See our weighted score and advice." (already in HTML)
    //    - "#final-score-gauge-fill" => color-coded fill from 0-100% (score 0–10)
    //-----------------------------------------------------------------------
    // (A) Score
    const scoreElem = document.getElementById('final-score');
    if (scoreElem) {
      const raw = (typeof finalData.total_score === 'number')
        ? finalData.total_score
        : 0;
      scoreElem.textContent = `Score: ${raw.toFixed(1)}`;
    }

    // (B) Visual gauge fill
    const gaugeFill = document.getElementById('final-score-gauge-fill');
    if (gaugeFill && typeof finalData.total_score === 'number') {
      // Clamp to 0..10 for a simple gauge
      const clampScore = Math.max(0, Math.min(10, finalData.total_score));
      const pct = (clampScore / 10) * 100;
      gaugeFill.style.width = pct + '%';

      // Color logic
      if (clampScore >= 7) {
        gaugeFill.style.backgroundColor = 'limegreen';
      } else if (clampScore >= 4) {
        gaugeFill.style.backgroundColor = 'orange';
      } else {
        gaugeFill.style.backgroundColor = 'red';
      }
    }

    //-----------------------------------------------------------------------
    // 3) Expanded State (#final-recommendation-full)
    //    Show factor-by-factor details from financial_analysis + sentiment,
    //    plus the final recommendation, rationale, and key_factors.
    //-----------------------------------------------------------------------
    const fullDiv = document.getElementById('final-recommendation-full');
    if (fullDiv) {
      // Pull out factor scores if present
      // (Ensure your backend sets finalData.factor_scores = { factor1_score, factor2_score, ... })
      const factor1Score = (finalData.factor_scores
        && typeof finalData.factor_scores.factor1_score === 'number')
          ? finalData.factor_scores.factor1_score
          : 'N/A';
      const factor2Score = (finalData.factor_scores
        && typeof finalData.factor_scores.factor2_score === 'number')
          ? finalData.factor_scores.factor2_score
          : 'N/A';
      const factor3Score = (finalData.factor_scores
        && typeof finalData.factor_scores.factor3_score === 'number')
          ? finalData.factor_scores.factor3_score
          : 'N/A';
      const factor4Score = (finalData.factor_scores
        && typeof finalData.factor_scores.factor4_score === 'number')
          ? finalData.factor_scores.factor4_score
          : 'N/A';
      const factor5Score = (finalData.factor_scores
        && typeof finalData.factor_scores.factor5_score === 'number')
          ? finalData.factor_scores.factor5_score
          : 'N/A';
      const factor6Score = (finalData.factor_scores
        && typeof finalData.factor_scores.factor6_score === 'number')
          ? finalData.factor_scores.factor6_score
          : 'N/A';

      // Retrieve financial analysis details
      const dcfVal = finAnalysisData.dcf_intrinsic_value || 0;
      // If your backend includes stock_price in the same object, adapt as needed:
      const stockPrice = (typeof finAnalysisData.stock_price === 'number')
        ? finAnalysisData.stock_price
        : 0;

      const ratioObj = finAnalysisData.ratios || {};
      const tsa = finAnalysisData.time_series_analysis || {};

      // Retrieve sentiment details
      const earnSent = sentimentData.earnings_call_sentiment || {};
      const indSent = sentimentData.industry_report_sentiment || {};
      const ecoSent = sentimentData.economic_report_sentiment || {};

      // Decide recommendation color
      let recColor = 'limegreen';
      if (finalData.recommendation
          && finalData.recommendation.toUpperCase() === 'SELL') {
        recColor = 'red';
      } else if (finalData.recommendation
          && finalData.recommendation.toUpperCase() === 'HOLD') {
        recColor = 'orange';
      }

      // Build final HTML for expanded
      let expandedHTML = `
        <h3>Financial Analysis</h3>

        <h4>Discounted Cash Flow (DCF) Analysis</h4>
        <ul>
          <li>Intrinsic Value per Share: <strong>$${dcfVal.toFixed(2)}</strong></li>
          <li>Current Stock Price: <strong>$${stockPrice.toFixed(2)}</strong></li>
          <li>Factor 1 Score (DCF Analysis): <strong>${factor1Score}</strong></li>
        </ul>

        <h4>Ratio Analysis</h4>
        <table class="table table-sm">
          <thead>
            <tr><th>Ratio</th><th>Value</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Debt-to-Equity Ratio</td>
              <td>${
                (typeof ratioObj["Debt-to-Equity Ratio"] === 'number')
                  ? ratioObj["Debt-to-Equity Ratio"].toFixed(2)
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>Current Ratio</td>
              <td>${
                (typeof ratioObj["Current Ratio"] === 'number')
                  ? ratioObj["Current Ratio"].toFixed(2)
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>P/E Ratio</td>
              <td>${
                (typeof ratioObj["P/E Ratio"] === 'number')
                  ? ratioObj["P/E Ratio"].toFixed(2)
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>P/B Ratio</td>
              <td>${
                (typeof ratioObj["P/B Ratio"] === 'number')
                  ? ratioObj["P/B Ratio"].toFixed(2)
                  : 'N/A'
              }</td>
            </tr>
          </tbody>
        </table>
        <p>Factor 2 Score (Ratio Analysis): <strong>${factor2Score}</strong></p>

        <h4>Time Series Analysis</h4>
        <table class="table table-sm">
          <thead>
            <tr><th>Metric</th><th>CAGR</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Revenue CAGR</td>
              <td>${
                (typeof tsa.revenue_cagr === 'number')
                  ? (tsa.revenue_cagr * 100).toFixed(2) + '%'
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>Net Income CAGR</td>
              <td>${
                (typeof tsa.net_income_cagr === 'number')
                  ? (tsa.net_income_cagr * 100).toFixed(2) + '%'
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>Total Assets CAGR</td>
              <td>${
                (typeof tsa.assets_cagr === 'number')
                  ? (tsa.assets_cagr * 100).toFixed(2) + '%'
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>Total Liabilities CAGR</td>
              <td>${
                (typeof tsa.liabilities_cagr === 'number')
                  ? (tsa.liabilities_cagr * 100).toFixed(2) + '%'
                  : 'N/A'
              }</td>
            </tr>
            <tr>
              <td>Operating Cash Flow CAGR</td>
              <td>${
                (typeof tsa.cashflow_cagr === 'number')
                  ? (tsa.cashflow_cagr * 100).toFixed(2) + '%'
                  : 'N/A'
              }</td>
            </tr>
          </tbody>
        </table>
        <p>Factor 3 Score (Time Series Analysis): <strong>${factor3Score}</strong></p>

        <h3>Sentiment Analysis</h3>

        <h5>Earnings Call Sentiment</h5>
        <ul>
          <li>Sentiment Score: ${
            (typeof earnSent.score === 'number')
              ? earnSent.score.toFixed(2)
              : 'N/A'
          }</li>
          <li>Factor 4 Score: <strong>${factor4Score}</strong></li>
        </ul>

        <h5>Industry Report Sentiment</h5>
        <ul>
          <li>Sentiment Score: ${
            (typeof indSent.score === 'number')
              ? indSent.score.toFixed(2)
              : 'N/A'
          }</li>
          <li>Factor 5 Score: <strong>${factor5Score}</strong></li>
        </ul>

        <h5>Economic Report Sentiment</h5>
        <ul>
          <li>Sentiment Score: ${
            (typeof ecoSent.score === 'number')
              ? ecoSent.score.toFixed(2)
              : 'N/A'
          }</li>
          <li>Factor 6 Score: <strong>${factor6Score}</strong></li>
        </ul>

        <h3>Final Recommendation</h3>
        <p>The weighted total score based on the analysis is:
          <strong>${
            (typeof finalData.total_score === 'number')
              ? finalData.total_score.toFixed(1)
              : 'N/A'
          }</strong>.</p>
        <p>The final recommendation is:
          <strong>${
            (finalData.recommendation)
              ? finalData.recommendation
              : 'N/A'
          }</strong>.</p>
        <p><strong>Rationale:</strong> ${
          finalData.rationale
            ? finalData.rationale
            : 'No rationale provided.'
        }</p>
        <ul>
          ${
            (Array.isArray(finalData.key_factors))
              ? finalData.key_factors.map(kf => `<li>${kf}</li>`).join('')
              : ''
          }
        </ul>
        <div style="margin-top:10px;">
          <strong>Visual:</strong>
          <span style="
            display:inline-block;
            width:60px;
            height:20px;
            border-radius:4px;
            margin-left:8px;
            background:${recColor};
          "></span>
        </div>
      `;

      // Insert into the expanded area
      fullDiv.innerHTML = expandedHTML;
    }
  })
  .catch(err => {
    console.error("Error fetching final recommendation:", err);
  });


  // (F) Chatbot - Not doing any default fetch here.

  // (G) Scenario Analysis - triggered by user.

  // (H) General Company Info
  let storedSector = null;

  fetch('/system1/company_info_data')
    .then((res) => {
      if (res.ok) return res.json();
      throw new Error('Failed to fetch company info');
    })
    .then((basicData) => {
      // Save the sector in a variable so we can use it after the second fetch
      storedSector = basicData.sector || 'Tech';

      // Collapsed state => show only the sector
      const collapsedKey = document.querySelector('#company-info-container .state-collapsed .tiny-data');
      if (collapsedKey) {
        collapsedKey.textContent = `Sector: ${storedSector}`;
      }

      // Now fetch the GPT-based details (c_suite + analysis) before handling default + expanded
      return fetch('/system1/company_info_details');
    })
    .then((res) => {
      if (res.ok) return res.json();
      throw new Error('Failed to fetch GPT-based company info');
    })
    .then((gptData) => {
      // Default State => "Sector: X" + "Key Executives: Y"
      const defaultKey = document.querySelector('#company-info-container .state-default .key-metric');
      if (defaultKey) {
        defaultKey.innerHTML = `
          Sector: ${storedSector} <br />
          Key Executives: ${gptData.c_suite || 'N/A'}
        `;
      }

      // Expanded State => show everything (sector, c-suite, and analysis)
      const fullDiv = document.getElementById('company-info-full');
      if (fullDiv) {
        // Let’s do a simple JSON stringify here:
        const analysisJsonString = JSON.stringify(gptData.analysis, null, 2);

        fullDiv.innerHTML = `
          <p><strong>Sector:</strong> ${storedSector}</p>
          <p><strong>C-suite Executives:</strong> ${gptData.c_suite || 'N/A'}</p>
          <p><strong>GPT-based Analysis:</strong></p>
          <pre>${analysisJsonString}</pre>
        `;
      }
    })
    .catch((err) => {
      console.error('Error fetching company info:', err);
    });

  // (I) Populate the S3 doc dropdown (Renamed to chat-document-select)
  fetch('/system2/list_documents')
    .then(res => res.json())
    .then(docKeys => {
      const docDropdown = document.getElementById('chat-document-select'); // <-- RENAMED
      if (!docDropdown) return;

      // Clear existing options
      while (docDropdown.firstChild) {
        docDropdown.removeChild(docDropdown.firstChild);
      }

      // Placeholder
      const placeholderOpt = document.createElement('option');
      placeholderOpt.value = '';
      placeholderOpt.textContent = '(Choose a PDF from S3)';
      docDropdown.appendChild(placeholderOpt);

      docKeys.forEach(key => {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = key;
        docDropdown.appendChild(option);
      });
    })
    .catch(err => {
      console.error("Error fetching S3 document list:", err);
    });

 // Finally, load Data Visualizations
 initDataVisualizations();
}

// ---------------------------------------------------------------------------
// 2. Chatbot Functionality
// ---------------------------------------------------------------------------
function initChatbot() {
  const chatSendBtn = document.getElementById('chat-send');
  if (!chatSendBtn) return;

  chatSendBtn.addEventListener('click', function() {
    const messageInput = document.getElementById('chat-input');
    const chatResponse = document.getElementById('chat-response');
    const docDropdown = document.getElementById('chat-document-select'); // <-- RENAMED

    if (!messageInput || !chatResponse || !docDropdown) return;

    const message = messageInput.value.trim();
    if (!message) return;

    const documentId = docDropdown.value || '';

    fetch('/system2/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        document_id: documentId
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.answer) {
        chatResponse.textContent = data.answer;
      } else if (data.error) {
        chatResponse.textContent = data.error;
      } else {
        chatResponse.textContent = 'No response received.';
      }
    })
    .catch(err => {
      chatResponse.textContent = 'Error communicating with chatbot.';
      console.error(err);
    });
  });
}

// main.js
// ---------------------------------------------------------------------------
// A global variable to hold the user-chosen ticker from ?ticker=XYZ
// ---------------------------------------------------------------------------
let userTicker = "MSFT"; // fallback if the URL param is missing

// ---------------------------------------------------------------------------
// Read the ticker from the query params as soon as DOM is ready
// Then initialize base-case logic & scenario logic
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // 1) Parse ?ticker= from the URL
  const urlParams = new URLSearchParams(window.location.search);
  userTicker = urlParams.get("ticker") || "MSFT";
  console.log("Parsed userTicker from URL:", userTicker);

  // 2) Init Base Case
  initBaseCase();

  // 3) Init Scenario Analysis
  initScenarioAnalysis();
});

// ---------------------------------------------------------------------------
// NEW: Confirm the frontend calls /system3/base_case to load the agentic base case
// ---------------------------------------------------------------------------
function initBaseCase() {
  // Use the user-chosen ticker from userTicker
  const endpoint = `/system3/base_case?ticker=${userTicker}`;
  console.log("Calling base_case endpoint:", endpoint);

  fetch(endpoint)
    .then(res => res.json())
    .then(data => {
      console.log("Base Case from system3:", data);
      // Suppose we display the base-case intrinsic value in some element
      const baseCaseElem = document.getElementById('base-case-output');
      if (baseCaseElem && data.intrinsic_value_per_share) {
        baseCaseElem.textContent =
          "Base Case IV: $" + data.intrinsic_value_per_share.toFixed(2);
      } else if (baseCaseElem) {
        // If there's an error or no data, handle gracefully
        if (data.error) {
          baseCaseElem.textContent = "Error: " + data.error;
        } else {
          baseCaseElem.textContent = "No data returned from /system3/base_case.";
        }
      }
    })
    .catch(err => {
      console.error("Error fetching base case from system3:", err);
      const baseCaseElem = document.getElementById('base-case-output');
      if (baseCaseElem) {
        baseCaseElem.textContent = "Error fetching base case.";
      }
    });
}

// ---------------------------------------------------------------------------
// 3. Scenario Analysis
// ---------------------------------------------------------------------------
function initScenarioAnalysis() {
  const scenarioBtn = document.getElementById('run-scenario-btn');
  if (!scenarioBtn) return;

  scenarioBtn.addEventListener('click', function() {
    const scenario = document.getElementById('scenario-select').value;
    const wacc = parseFloat(document.getElementById('wacc-input').value) || 10;
    const revGrowth = parseFloat(document.getElementById('rev-growth-input').value) || 5;
    const opex = parseFloat(document.getElementById('opex-input').value) || 20;
    const cogs = parseFloat(document.getElementById('cogs-input').value) || 60;
    const taxRate = parseFloat(document.getElementById('tax-rate-input').value) || 21;

    const scenarioIntrinsicElem = document.getElementById('scenario-intrinsic-value');
    const scenarioChartElem = document.getElementById('scenario-chart');

    // Instead of legacy CSV-based route, call /system3/calculate_alpha
    // We'll pass minimal fields: ticker from userTicker, wacc, etc.
    const payload = {
      ticker: userTicker,  // use userTicker, not 'MSFT'
      scenario: scenario,
      wacc: wacc / 100,
      revenue_growth_rate: revGrowth / 100,
      operating_expenses_pct: opex / 100,
      cogs_pct: cogs / 100,
      tax_rate: taxRate / 100
    };

    console.log("Sending scenario payload to /system3/calculate_alpha:", payload);

    fetch('/system3/calculate_alpha', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      console.log("Scenario response from system3:", data);
      if (data.intrinsic_value_per_share) {
        scenarioIntrinsicElem.textContent = data.intrinsic_value_per_share.toFixed(2);
      } else if (data.error) {
        scenarioIntrinsicElem.textContent = data.error;
      } else {
        scenarioIntrinsicElem.textContent = 'No scenario data received.';
      }
    })
    .catch(err => {
      scenarioIntrinsicElem.textContent = 'Error retrieving scenario data.';
      console.error(err);
    });
  });
}

// ---------------------------------------------------------------------------
// 4. Three-Row Tile Layout
//    Toggles .collapsed for non-expanded tiles, .enlarged for the clicked tile,
//    and toggles .expanded-layout on the #dashboardGrid parent for 3-row layout
// ---------------------------------------------------------------------------
function initThreeRowTileLayout() {
  const tiles = Array.from(document.querySelectorAll('.dashboard-tile'));
  const customizeTile = document.getElementById('tile-customize-dashboard');

  // 8 main + 1 "Customize"
  const tileOrder = [
    'company-report-container',
    'financial-analysis-container',
    'sentiment-analysis-container',
    'data-visualizations-container',
    'final-recommendation-container',
    'chatbot-container',
    'scenario-analysis-container',
    'company-info-container',
    'tile-customize-dashboard'
  ];

  let currentlyExpandedTileId = null;

  tileOrder.forEach(id => {
    if (id === 'tile-customize-dashboard') return;
    const tile = document.getElementById(id);
    if (!tile) return;

    const header = tile.querySelector('.card-header');
    if (header) {
      header.addEventListener('click', () => expandTileInMiddle(id));
    }
  });

  function expandTileInMiddle(tileId) {
    if (currentlyExpandedTileId === tileId) {
      collapseAllToDefault();
      currentlyExpandedTileId = null;
      return;
    }
    currentlyExpandedTileId = tileId;

    const grid = document.getElementById('dashboardGrid');
    grid.classList.add('expanded-layout');

    const normalTiles = tileOrder.filter(id => id !== 'tile-customize-dashboard' && id !== tileId);
    const topTileIds = normalTiles.slice(0, 4);
    const bottomTileIds = normalTiles.slice(4);

    topTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove('expanded','middle-row','bottom-row','hidden','minimized','enlarged');
      t.classList.add('top-row','minimized','collapsed');
    });

    const middleTile = document.getElementById(tileId);
    middleTile.classList.remove('top-row','bottom-row','hidden','minimized','collapsed');
    middleTile.classList.add('middle-row','expanded','enlarged');

    bottomTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove('expanded','middle-row','top-row','hidden','minimized','enlarged');
      t.classList.add('bottom-row','collapsed');
    });

    if (customizeTile) {
      customizeTile.classList.remove('hidden');
      customizeTile.classList.remove('top-row','middle-row','enlarged','expanded');
      customizeTile.classList.add('bottom-row','collapsed');
    }
  }

  function collapseAllToDefault() {
    const grid = document.getElementById('dashboardGrid');
    grid.classList.remove('expanded-layout');

    tileOrder.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove(
        'expanded','middle-row','bottom-row','hidden','minimized','collapsed','enlarged','top-row'
      );
    });
    if (customizeTile) customizeTile.classList.add('hidden');
  }
}
