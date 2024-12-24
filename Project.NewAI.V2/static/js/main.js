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
//
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  // 1. Load data into each tile
  initDashboard();

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
  // -------------------------------------------------------------------------
  // (A) Company Report
  // -------------------------------------------------------------------------
  fetch('/system1/company_report_data')
    .then(res => res.json())
    .then(data => {
      // 1) Insert short snippet => default & collapsed
      const defaultKey = document.querySelector('#company-report-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#company-report-container .state-collapsed .tiny-data');
      if (defaultKey) defaultKey.textContent = data.stock_price || '$145.32';
      if (collapsedKey) collapsedKey.textContent = data.stock_price || '$145.32';

      // 2) Insert full analysis => enlarged
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

  // -------------------------------------------------------------------------
  // (B) Financial Analysis
  // -------------------------------------------------------------------------
  fetch('/system1/financial_analysis_data')
    .then(res => res.json())
    .then(data => {
      // short snippet
      const dcfValueDefault = document.querySelector('#financial-analysis-container .state-default .key-metric');
      const dcfValueCollapsed = document.querySelector('#financial-analysis-container .state-collapsed .tiny-data');
      // e.g. $127.50 if not found in data
      const dcfStr = data.dcf_intrinsic_value ? `$${data.dcf_intrinsic_value.toFixed(2)}` : '$127.50';
      if (dcfValueDefault) dcfValueDefault.textContent = dcfStr;
      if (dcfValueCollapsed) dcfValueCollapsed.textContent = dcfStr;

      // full analysis
      const fullDiv = document.getElementById('financial-analysis-full');
      if (fullDiv) {
        fullDiv.innerHTML = `
          <p>DCF Intrinsic Value: ${dcfStr}</p>
          <p>Key Ratios: ${JSON.stringify(data.ratios)}</p>
          <p>Time Series Analysis: ${JSON.stringify(data.time_series_analysis)}</p>
        `;
      }
    })
    .catch(err => {
      console.error("Error fetching financial analysis data:", err);
    });

  // -------------------------------------------------------------------------
  // (C) Sentiment Analysis
  // -------------------------------------------------------------------------
  fetch('/system1/sentiment_data')
    .then(res => res.json())
    .then(data => {
      // default + collapsed
      const defaultKey = document.querySelector('#sentiment-analysis-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#sentiment-analysis-container .state-collapsed .tiny-data');
      // Suppose data has a .composite_score or fallback
      const compositeScore = (data.composite_score !== undefined) ? data.composite_score : '+0.25';
      if (defaultKey) defaultKey.textContent = compositeScore;
      if (collapsedKey) collapsedKey.textContent = compositeScore;

      // enlarged
      const fullDiv = document.getElementById('sentiment-analysis-full');
      if (fullDiv) {
        // Using the existing data for each sentiment
        const earnings = data.earnings_call_sentiment;
        const industry = data.industry_report_sentiment;
        const economic = data.economic_report_sentiment;
        fullDiv.innerHTML = `
          <p>Earnings Call Sentiment: Score ${earnings.score}, ${earnings.explanation}</p>
          <p>Industry Report Sentiment: Score ${industry.score}, ${industry.explanation}</p>
          <p>Economic Report Sentiment: Score ${economic.score}, ${economic.explanation}</p>
        `;
      }
    })
    .catch(err => {
      console.error("Error fetching sentiment data:", err);
    });

  // -------------------------------------------------------------------------
  // (D) Data Visualizations
  // -------------------------------------------------------------------------
  fetch('/system1/data_visualizations_data')
    .then(res => res.json())
    .then(data => {
      // default + collapsed
      const defaultKey = document.querySelector('#data-visualizations-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#data-visualizations-container .state-collapsed .tiny-data');
      // e.g. data.latest_revenue or fallback
      if (defaultKey) defaultKey.textContent = data.latest_revenue || 'Q2: $50.5B';
      if (collapsedKey) collapsedKey.textContent = data.latest_revenue || '$50.5B';

      // enlarged
      const fullDiv = document.getElementById('data-visualizations-full');
      if (fullDiv) {
        fullDiv.innerHTML = `
          <p>Data for visualizations loaded. Below is raw JSON:</p>
          <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
      }
    })
    .catch(err => {
      console.error("Error fetching data visualizations:", err);
    });

  // -------------------------------------------------------------------------
  // (E) Final Recommendation
  // -------------------------------------------------------------------------
  fetch('/system1/final_recommendation')
    .then(res => res.json())
    .then(data => {
      // default + collapsed
      const recDefault = document.querySelector('#final-recommendation-container .state-default .key-metric');
      const recCollapsed = document.querySelector('#final-recommendation-container .state-collapsed .tiny-data');
      if (recDefault) recDefault.textContent = data.recommendation || 'BUY';
      if (recCollapsed) recCollapsed.textContent = data.recommendation || 'BUY';

      // enlarged
      const fullDiv = document.getElementById('final-recommendation-full');
      if (fullDiv) {
        fullDiv.innerHTML = `
          <p>The weighted total score: ${data.total_score}</p>
          <p>Final Recommendation: ${data.recommendation}</p>
        `;
      }
    })
    .catch(err => {
      console.error("Error fetching final recommendation:", err);
    });

  // -------------------------------------------------------------------------
  // (F) Chatbot
  // Typically, the Chatbot tile is interactive, so no big data fetch needed.
  // If you wanted to show a snippet from the server, you could do so here.

  // -------------------------------------------------------------------------
  // (G) Scenario Analysis
  // The user triggers scenario calculations manually, so no default fetch needed.
  // You can do it if you want a baseline scenario, but not mandatory.

  // -------------------------------------------------------------------------
  // (H) General Company Info
  // -------------------------------------------------------------------------
  fetch('/system1/company_info_data')
    .then(res => {
      if (res.ok) return res.json();
      throw new Error('Failed to fetch company info');
    })
    .then(data => {
      // default + collapsed
      const defaultKey = document.querySelector('#company-info-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#company-info-container .state-collapsed .tiny-data');
      if (defaultKey) defaultKey.textContent = `Sector: ${data.sector || 'Tech'}`;
      if (collapsedKey) collapsedKey.textContent = data.sector || 'Tech';

      // enlarged
      const fullDiv = document.getElementById('company-info-full');
      if (fullDiv) {
        fullDiv.innerHTML = `
          <p>C-suite Executives: ${data.c_suite_executives}</p>
          <p>Shares Outstanding: ${data.shares_outstanding}</p>
          <p>WACC: ${data.wacc}</p>
          <p>P/E: ${data.pe_ratio}</p>
          <p>P/S: ${data.ps_ratio}</p>
          <p>Sector, Industry, Sub-industry: 
             ${data.sector}, ${data.industry}, ${data.sub_industry}</p>
        `;
      }
    })
    .catch(err => {
      console.error("Error fetching company info:", err);
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
//    Toggles .collapsed for non-expanded tiles, .enlarged for the clicked tile
// ---------------------------------------------------------------------------
function initThreeRowTileLayout() {
  const tiles = Array.from(document.querySelectorAll('.dashboard-tile'));
  const customizeTile = document.getElementById('tile-customize-dashboard');

  // The order of tiles (8 main + 1 "Customize")
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

  // On page load, reset each tile to top-row (default state)
  tileOrder.forEach(id => {
    const t = document.getElementById(id);
    if (!t) return;
    t.classList.remove(
      'middle-row',
      'bottom-row',
      'expanded',
      'minimized',
      'hidden',
      'collapsed',
      'enlarged',
      'top-row'
    );
    t.classList.add('top-row');
  });
  if (customizeTile) customizeTile.classList.add('hidden');

  // For each main tile, attach a click event
  tileOrder.forEach(id => {
    if (id === 'tile-customize-dashboard') return; // skip "Customize"
    const tile = document.getElementById(id);
    tile.addEventListener('click', () => expandTileInMiddle(id));
  });

  let currentlyExpandedTileId = null;

  function expandTileInMiddle(tileId) {
    // If user clicks the tile that's already expanded => collapse
    if (currentlyExpandedTileId === tileId) {
      collapseAllToTop();
      currentlyExpandedTileId = null;
      return;
    }

    // Otherwise, user is expanding a new tile
    currentlyExpandedTileId = tileId;

    // define top row => first 4 (minus the clicked tile)
    const topTileIds = tileOrder.slice(0, 4).filter(x => x !== tileId);
    const middleTileId = tileId;
    // define bottom row => last 4 (plus "Customize")
    const bottomTileIds = tileOrder.slice(4);
    // remove the clicked tile from bottom row if present
    const idx = bottomTileIds.indexOf(tileId);
    if (idx >= 0) bottomTileIds.splice(idx, 1);

    // 1) Move topTileIds => row 1 (collapsed, minimized)
    topTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove(
        'expanded',
        'middle-row',
        'bottom-row',
        'hidden',
        'minimized',
        'enlarged'
      );
      t.classList.add('top-row', 'minimized', 'collapsed');
    });

    // 2) Middle tile => row 2, expanded + enlarged
    const middleTile = document.getElementById(middleTileId);
    middleTile.classList.remove(
      'top-row',
      'bottom-row',
      'hidden',
      'minimized',
      'collapsed'
    );
    middleTile.classList.add('middle-row', 'expanded', 'enlarged');

    // 3) Bottom row => the rest
    bottomTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove(
        'expanded',
        'middle-row',
        'top-row',
        'hidden',
        'minimized',
        'enlarged'
      );
      t.classList.add('bottom-row', 'collapsed');
    });

    // Un-hide "Customize" tile if present
    if (customizeTile) customizeTile.classList.remove('hidden');
  }

  function collapseAllToTop() {
    tileOrder.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove(
        'expanded',
        'middle-row',
        'bottom-row',
        'hidden',
        'minimized',
        'collapsed',
        'enlarged'
      );
      t.classList.add('top-row');
    });
    if (customizeTile) customizeTile.classList.add('hidden');
  }
}
