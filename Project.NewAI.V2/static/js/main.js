// main.js
// ---------------------------------------------------------------------------
// This file handles all AJAX requests to the backend and updates the dashboard
// with real-time data. It also includes chatbot and scenario analysis features.
//
// NEW: We have replaced the old tile expansion logic with a 3-row layout approach:
//   - 4 tiles in the top row
//   - 1 expanded tile in the middle row
//   - 4 tiles in the bottom row (including "Customize Dashboard")
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  // 1. Data fetching for the dashboard
  initDashboard();

  // 2. Chatbot functionality
  initChatbot();

  // 3. Scenario Analysis functionality
  initScenarioAnalysis();

  // 4. Three-row tile layout logic
  initThreeRowTileLayout();
});

// ---------------------------------------------------------------------------
// 1. Data Fetching + Populating Dashboard
// ---------------------------------------------------------------------------
function initDashboard() {
  // 1a. Fetch and populate the Company Report
  fetch('/system1/company_report_data')
    .then(response => response.json())
    .then(data => {
      document.querySelector('#company-report-container .card-body').innerHTML = `
        <p><strong>Executive Summary:</strong> ${data.executive_summary}</p>
        <p><strong>Company Summary:</strong> ${data.company_summary}</p>
        <p><strong>Industry Summary:</strong> ${data.industry_summary}</p>
        <p><strong>Risk Considerations:</strong> ${data.risk_considerations}</p>
      `;
      document.getElementById('exec-summary').textContent = data.executive_summary;
      document.getElementById('comp-summary').textContent = data.company_summary;
      document.getElementById('ind-summary').textContent = data.industry_summary;
      document.getElementById('risk-considerations').textContent = data.risk_considerations;
    })
    .catch(err => {
      console.error("Error fetching company report data:", err);
      document.querySelector('#company-report-container .card-body').innerHTML =
        '<p>Failed to load company report data.</p>';
    });

  // 1b. Fetch and populate Financial Analysis
  fetch('/system1/financial_analysis_data')
    .then(res => res.json())
    .then(data => {
      document.querySelector('#financial-analysis-container .card-body').innerHTML = `
        <p>DCF Intrinsic Value: $${data.dcf_intrinsic_value.toFixed(2)}</p>
        <p>Key Ratios: ${JSON.stringify(data.ratios)}</p>
        <p>Time Series Analysis: ${JSON.stringify(data.time_series_analysis)}</p>
      `;
      document.getElementById('dcf-value').textContent = `$${data.dcf_intrinsic_value.toFixed(2)}`;
      document.getElementById('ratio-analysis').textContent = JSON.stringify(data.ratios);
      document.getElementById('time-series').textContent = JSON.stringify(data.time_series_analysis);
    })
    .catch(err => {
      console.error("Error fetching financial analysis data:", err);
      document.querySelector('#financial-analysis-container .card-body').innerHTML =
        '<p>Failed to load financial analysis data.</p>';
    });

  // 1c. Fetch and populate Sentiment Analysis
  fetch('/system1/sentiment_data')
    .then(res => res.json())
    .then(data => {
      const earnings = data.earnings_call_sentiment;
      const industry = data.industry_report_sentiment;
      const economic = data.economic_report_sentiment;

      document.querySelector('#sentiment-analysis-container .card-body').innerHTML = `
        <p>Earnings Call Sentiment: Score ${earnings.score}, ${earnings.explanation}</p>
        <p>Industry Report Sentiment: Score ${industry.score}, ${industry.explanation}</p>
        <p>Economic Report Sentiment: Score ${economic.score}, ${economic.explanation}</p>
      `;

      document.getElementById('earnings-sentiment').textContent =
        `Score ${earnings.score}, ${earnings.explanation}`;
      document.getElementById('industry-sentiment').textContent =
        `Score ${industry.score}, ${industry.explanation}`;
      document.getElementById('economic-sentiment').textContent =
        `Score ${economic.score}, ${economic.explanation}`;
    })
    .catch(err => {
      console.error("Error fetching sentiment data:", err);
      document.querySelector('#sentiment-analysis-container .card-body').innerHTML =
        '<p>Failed to load sentiment data.</p>';
    });

  // 1d. Fetch Data Visualizations Data
  fetch('/system1/data_visualizations_data')
    .then(res => res.json())
    .then(data => {
      document.querySelector('#data-visualizations-container .card-body').innerHTML = `
        <p>Data for visualizations loaded. Below is raw JSON:</p>
        <pre>${JSON.stringify(data, null, 2)}</pre>
      `;
      // (Optional) Initialize charts with 'data' here
    })
    .catch(err => {
      console.error("Error fetching data visualizations:", err);
      document.querySelector('#data-visualizations-container .card-body').innerHTML =
        '<p>Failed to load visualization data.</p>';
    });

  // 1e. Fetch Final Recommendation
  fetch('/system1/final_recommendation')
    .then(res => res.json())
    .then(data => {
      document.querySelector('#final-recommendation-container .card-body').innerHTML = `
        <p>The weighted total score: ${data.total_score}</p>
        <p>Final Recommendation: ${data.recommendation}</p>
      `;
      document.getElementById('total-score').textContent = data.total_score;
      document.getElementById('recommendation').textContent = data.recommendation;
    })
    .catch(err => {
      console.error("Error fetching final recommendation:", err);
      document.querySelector('#final-recommendation-container .card-body').innerHTML =
        '<p>Failed to load recommendation.</p>';
    });

  // 1f. Fetch General Company Info
  fetch('/system1/company_info_data')
    .then(res => {
      if (res.ok) return res.json();
      throw new Error('Failed to fetch company info');
    })
    .then(data => {
      document.querySelector('#company-info-container .card-body').innerHTML = `
        <p>C-suite Executives: ${data.c_suite_executives}</p>
        <p>Shares Outstanding: ${data.shares_outstanding}</p>
        <p>WACC: ${data.wacc}</p>
        <p>P/E: ${data.pe_ratio}</p>
        <p>P/S: ${data.ps_ratio}</p>
        <p>Sector, Industry, Sub-industry: ${data.sector}, ${data.industry}, ${data.sub_industry}</p>
      `;
      document.getElementById('c-suite').textContent = data.c_suite_executives;
      document.getElementById('shares-outstanding').textContent = data.shares_outstanding;
      document.getElementById('info-wacc').textContent = data.wacc;
      document.getElementById('pe-ratio').textContent = data.pe_ratio;
      document.getElementById('ps-ratio').textContent = data.ps_ratio;
      document.getElementById('sector-info').textContent =
        `${data.sector}, ${data.industry}, ${data.sub_industry}`;
    })
    .catch(err => {
      console.error("Error fetching company info:", err);
      document.querySelector('#company-info-container .card-body').innerHTML =
        '<p>Failed to load company info.</p>';
    });
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
    if (!messageInput || !chatResponse) return;

    const message = messageInput.value.trim();
    if (!message) return;

    fetch('/system2/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        document_id: DOCUMENT_ID
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

    const payload = {
      sector: SECTOR,
      industry: INDUSTRY,
      sub_industry: SUB_INDUSTRY,
      scenario: scenario,
      stock_ticker: STOCK_TICKER,
      data: {
        income_statement: {},
        balance_sheet: {},
        cash_flow_statement: {}
      },
      assumptions: {
        wacc: wacc / 100,
        revenue_growth_rate: revGrowth / 100,
        operating_expenses_pct: opex / 100,
        cogs_pct: cogs / 100,
        tax_rate: taxRate / 100
      }
    };

    fetch('/system3/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      if (data.intrinsic_value_per_share) {
        scenarioIntrinsicElem.textContent = data.intrinsic_value_per_share.toFixed(2);
        // If the response includes additional data, it can be displayed or graphed here
        // e.g., scenarioChartElem.innerHTML = ...
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
//    When a tile is clicked, it expands to row 2, while the top 4 go to row 1
//    and the bottom 4 go to row 3 (including "Customize Dashboard").
// ---------------------------------------------------------------------------
function initThreeRowTileLayout() {
  // Grab references to all tiles (including "Customize Dashboard")
  const tiles = Array.from(document.querySelectorAll('.dashboard-tile'));
  const customizeTile = document.getElementById('tile-customize-dashboard');

  // We'll define a consistent order for the 9 tiles (8 main + 1 customize).
  const tileOrder = [
    'company-report-container',
    'financial-analysis-container',
    'sentiment-analysis-container',
    'data-visualizations-container',
    'final-recommendation-container',
    'chatbot-container',
    'scenario-analysis-container',
    'company-info-container',
    'tile-customize-dashboard' // the "Customize" tile
  ];

  // Initially, let's place all 8 standard tiles in the top row, hide "Customize".
  tileOrder.forEach(id => {
    const t = document.getElementById(id);
    if (!t) return;
    // Remove any leftover classes from a previous session
    t.classList.remove('middle-row', 'bottom-row', 'expanded', 'minimized', 'hidden', 'top-row');
    // Add top-row
    t.classList.add('top-row');
  });
  // Hide the Customize tile (if it exists)
  if (customizeTile) customizeTile.classList.add('hidden');

  // For each of the 8 main tiles (not the customize tile), attach a click event
  tileOrder.forEach(id => {
    if (id === 'tile-customize-dashboard') return; // skip the "Customize" tile
    const tile = document.getElementById(id);
    tile.addEventListener('click', () => expandTileInMiddle(id));
  });

  let currentlyExpandedTileId = null;

  function expandTileInMiddle(tileId) {
    // If user clicks the same tile that is expanded, collapse everything
    if (currentlyExpandedTileId === tileId) {
      collapseAllToTop();
      currentlyExpandedTileId = null;
      return;
    }

    // Otherwise, user is expanding a new tile
    currentlyExpandedTileId = tileId;

    // We want 4 in top row, 1 in middle row (expanded), 4 in bottom row
    // We'll define top row as the first 4 in tileOrder (except the clicked tile).
    // The clicked tile is the middle row, expanded.
    // The last 4 in tileOrder (plus "Customize") go on bottom row,
    // with "Customize" un-hidden.

    const topTileIds = tileOrder.slice(0, 4).filter(x => x !== tileId);
    const middleTileId = tileId;
    const bottomTileIds = tileOrder.slice(4); // last 5 or so (including "Customize")
    // If the clicked tile is among them, remove it
    const idx = bottomTileIds.indexOf(tileId);
    if (idx >= 0) bottomTileIds.splice(idx, 1);

    // 1) Move topTileIds to row 1
    topTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove('expanded', 'middle-row', 'bottom-row', 'hidden');
      t.classList.add('top-row');
    });

    // 2) Middle tile => row 2, expanded
    const middleTile = document.getElementById(middleTileId);
    middleTile.classList.remove('top-row', 'bottom-row', 'hidden');
    middleTile.classList.add('middle-row', 'expanded');

    // 3) Bottom row => the remaining + "Customize"
    bottomTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove('expanded', 'middle-row', 'top-row', 'hidden');
      t.classList.add('bottom-row');
    });

    // Un-hide the "Customize Dashboard" tile (if present)
    if (customizeTile) customizeTile.classList.remove('hidden');
  }

  function collapseAllToTop() {
    tileOrder.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove('expanded', 'middle-row', 'bottom-row', 'hidden');
      t.classList.add('top-row');
    });
    if (customizeTile) customizeTile.classList.add('hidden');
  }
}
