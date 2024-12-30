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
        fullDiv.innerHTML = 
          `<p><strong>Executive Summary:</strong> ${data.executive_summary}</p>
          <p><strong>Company Summary:</strong> ${data.company_summary}</p>
          <p><strong>Industry Summary:</strong> ${data.industry_summary}</p>
          <p><strong>Risk Considerations:</strong> ${data.risk_considerations}</p>`;
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
      const dcfStr = data.dcf_intrinsic_value ? `$${data.dcf_intrinsic_value.toFixed(2)}` : '$127.50';
      if (dcfValueDefault) dcfValueDefault.textContent = dcfStr;
      if (dcfValueCollapsed) dcfValueCollapsed.textContent = dcfStr;

      // full analysis
      const fullDiv = document.getElementById('financial-analysis-full');
      if (fullDiv) {
        fullDiv.innerHTML = 
          `<p>DCF Intrinsic Value: ${dcfStr}</p>
          <p>Key Ratios: ${JSON.stringify(data.ratios)}</p>
          <p>Time Series Analysis: ${JSON.stringify(data.time_series_analysis)}</p>`;
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
      const defaultKey = document.querySelector('#sentiment-analysis-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#sentiment-analysis-container .state-collapsed .tiny-data');
      const compositeScore = (data.composite_score !== undefined) ? data.composite_score : '+0.25';
      if (defaultKey) defaultKey.textContent = compositeScore;
      if (collapsedKey) collapsedKey.textContent = compositeScore;

      const fullDiv = document.getElementById('sentiment-analysis-full');
      if (fullDiv) {
        const earnings = data.earnings_call_sentiment;
        const industry = data.industry_report_sentiment;
        const economic = data.economic_report_sentiment;
        fullDiv.innerHTML = 
          `<p>Earnings Call Sentiment: Score ${earnings.score}, ${earnings.explanation}</p>
          <p>Industry Report Sentiment: Score ${industry.score}, ${industry.explanation}</p>
          <p>Economic Report Sentiment: Score ${economic.score}, ${economic.explanation}</p>`;
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
      const defaultKey = document.querySelector('#data-visualizations-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#data-visualizations-container .state-collapsed .tiny-data');
      if (defaultKey) defaultKey.textContent = data.latest_revenue || 'Q2: $50.5B';
      if (collapsedKey) collapsedKey.textContent = data.latest_revenue || '$50.5B';

      const fullDiv = document.getElementById('data-visualizations-full');
      if (fullDiv) {
        // Example: Chart.js usage
        fullDiv.innerHTML = '';

        const chartCanvas = document.createElement('canvas');
        chartCanvas.id = 'revenueChart';
        chartCanvas.width = 400;
        chartCanvas.height = 250;
        fullDiv.appendChild(chartCanvas);

        const ctx = chartCanvas.getContext('2d');
        const sampleLabels = (data.revenue_over_time || []).map(item => item.date || 'Unknown');
        const sampleValues = (data.revenue_over_time || []).map(item => item.value || 0);

        new Chart(ctx, {
          type: 'bar',
          data: {
            labels: sampleLabels.length ? sampleLabels : ['Q1','Q2','Q3','Q4'],
            datasets: [{
              label: 'Revenue',
              data: sampleValues.length ? sampleValues : [50, 60, 55, 70],
              backgroundColor: 'rgba(75, 192, 192, 0.4)'
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                beginAtZero: true
              }
            }
          }
        });
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
      const recDefault = document.querySelector('#final-recommendation-container .state-default .key-metric');
      const recCollapsed = document.querySelector('#final-recommendation-container .state-collapsed .tiny-data');
      if (recDefault) recDefault.textContent = data.recommendation || 'BUY';
      if (recCollapsed) recCollapsed.textContent = data.recommendation || 'BUY';

      const fullDiv = document.getElementById('final-recommendation-full');
      if (fullDiv) {
        fullDiv.innerHTML = 
          `<p>The weighted total score: ${data.total_score}</p>
          <p>Final Recommendation: ${data.recommendation}</p>
          <p><strong>Rationale:</strong> ${data.rationale || 'No rationale provided.'}</p>
          <ul>
            ${(data.key_factors || []).map(f => `<li>${f}</li>`).join('')}
          </ul>`;
      }
    })
    .catch(err => {
      console.error("Error fetching final recommendation:", err);
    });

  // -------------------------------------------------------------------------
  // (F) Chatbot
  // -------------------------------------------------------------------------
  // Typically, no big data fetch needed. If you want a snippet from the server,
  // you could do so here.

  // -------------------------------------------------------------------------
  // (G) Scenario Analysis
  // -------------------------------------------------------------------------
  // The user triggers scenario calculations manually.

  // -------------------------------------------------------------------------
  // (H) General Company Info
  // -------------------------------------------------------------------------
  fetch('/system1/company_info_data')
    .then(res => {
      if (res.ok) return res.json();
      throw new Error('Failed to fetch company info');
    })
    .then(basicData => {
      const defaultKey = document.querySelector('#company-info-container .state-default .key-metric');
      const collapsedKey = document.querySelector('#company-info-container .state-collapsed .tiny-data');
      if (defaultKey) defaultKey.textContent = `Sector: ${basicData.sector || 'Tech'}`;
      if (collapsedKey) collapsedKey.textContent = basicData.sector || 'Tech';

      // Return second fetch for GPT-based analysis
      return fetch('/system1/company_info_details');
    })
    .then(res => {
      if (res.ok) return res.json();
      throw new Error('Failed to fetch GPT-based company info');
    })
    .then(gptData => {
      const fullDiv = document.getElementById('company-info-full');
      if (fullDiv) {
        fullDiv.innerHTML = 
          `<p><strong>C-suite Executives:</strong> ${gptData.c_suite || 'N/A'}</p>
          <p><strong>GPT-based Analysis:</strong> ${gptData.analysis || 'No analysis available.'}</p>`;
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
//    Toggles .collapsed for non-expanded tiles, .enlarged for the clicked tile,
//    and toggles .expanded-layout on the #dashboardGrid parent for 3-row layout
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

  let currentlyExpandedTileId = null;

  // Attach click handlers ONLY to the .card-header, so the tile
  // doesn't instantly collapse when you click inside the body
  tileOrder.forEach(id => {
    if (id === 'tile-customize-dashboard') return; // skip "Customize"
    const tile = document.getElementById(id);
    if (!tile) return;

    const header = tile.querySelector('.card-header');
    if (header) {
      header.addEventListener('click', () => expandTileInMiddle(id));
    }
  });

  function expandTileInMiddle(tileId) {
    // If user clicks the tile that's already expanded => collapse
    if (currentlyExpandedTileId === tileId) {
      collapseAllToDefault();
      currentlyExpandedTileId = null;
      return;
    }

    currentlyExpandedTileId = tileId;

    // Switch the main grid to a 3-row layout
    const grid = document.getElementById('dashboardGrid');
    grid.classList.add('expanded-layout');

    // 1) Filter out the clicked tile + "Customize" from the normal 8.
    //    That leaves 7 tiles to split top/bottom.
    const normalTiles = tileOrder.filter(id => 
      id !== 'tile-customize-dashboard' && id !== tileId
    );

    // 2) First 4 => top row, next 3 => bottom row
    const topTileIds = normalTiles.slice(0, 4);
    const bottomTileIds = normalTiles.slice(4);

    // 3) Middle tile => the clicked tile
    const middleTileId = tileId;

    // 4) Top row => collapsed
    topTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove(
        'expanded','middle-row','bottom-row','hidden','minimized','enlarged'
      );
      t.classList.add('top-row','minimized','collapsed');
    });

    // 5) Middle row => expanded
    const middleTile = document.getElementById(middleTileId);
    middleTile.classList.remove(
      'top-row','bottom-row','hidden','minimized','collapsed'
    );
    middleTile.classList.add('middle-row','expanded','enlarged');

    // 6) Bottom row => collapsed
    bottomTileIds.forEach(id => {
      const t = document.getElementById(id);
      if (!t) return;
      t.classList.remove(
        'expanded','middle-row','top-row','hidden','minimized','enlarged'
      );
      t.classList.add('bottom-row','collapsed');
    });

    // 7) Un-hide "Customize" tile & place it in bottom row
    if (customizeTile) {
      customizeTile.classList.remove('hidden');
      customizeTile.classList.remove('top-row','middle-row','enlarged','expanded');
      customizeTile.classList.add('bottom-row','collapsed');
    }
  }

  function collapseAllToDefault() {
    // Remove 3-row layout => back to default 2×4
    const grid = document.getElementById('dashboardGrid');
    grid.classList.remove('expanded-layout');

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
        'enlarged',
        'top-row'
      );
    });
    if (customizeTile) customizeTile.classList.add('hidden');
  }
}
