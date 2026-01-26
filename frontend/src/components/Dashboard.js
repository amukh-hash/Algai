import React, { useState } from 'react';
import axios from 'axios';
import ConfidenceGauges from './ConfidenceGauges';
import GreeksRadar from './GreeksRadar';
import PnLCurve from './PnLCurve';
import './Dashboard.css';

const Dashboard = () => {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedTrade, setSelectedTrade] = useState(null);

  const runPipeline = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/v1/hybrid/run', {
        symbol: "AAPL",
        start_date: "2023-01-01",
        end_date: "2023-01-05"
      });
      setTrades(response.data);
      if(response.data.length > 0) setSelectedTrade(response.data[0]);
    } catch (error) {
      console.error("Error running pipeline:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header>
        <h2>Institutional Hybrid Engine</h2>
        <button onClick={runPipeline} disabled={loading}>
          {loading ? 'Running Async Pipe...' : 'Run Analysis'}
        </button>
      </header>

      <div className="glass-box-grid">
        {/* Column 1: The Why */}
        <div className="column">
          <h3>The Why (AI)</h3>
          {selectedTrade && (
            <ConfidenceGauges 
              direction={selectedTrade.direction_prob || 0.5} 
              volatility={selectedTrade.vol_pred || 0.0} 
            />
          )}
          {/* Placeholder for Attention Heatmap */}
          <div className="card placeholder-heatmap">Attention Heatmap (Mock)</div>
        </div>

        {/* Column 2: The What */}
        <div className="column">
          <h3>The What (Trade)</h3>
          {selectedTrade && (
            <>
              <div className="card">
                <h4>Selected Contract</h4>
                <p><strong>{selectedTrade.root}</strong> {selectedTrade.expiration}</p>
                <p>Strike: {selectedTrade.strike} {selectedTrade.right}</p>
                <p>Entry: ${selectedTrade.entry_price}</p>
              </div>
              <GreeksRadar greeks={{
                delta: 0.4, // In real app, pass from trade data
                gamma: 0.05,
                theta: -0.02,
                vega: 0.1
              }} />
            </>
          )}
        </div>

        {/* Column 3: The How */}
        <div className="column">
          <h3>The How (Lifecycle)</h3>
          <PnLCurve data={trades.map(t => ({ date: t.entry_date, pnl: t.pnl }))} />
          <div className="card">
            <h4>Trade Log</h4>
            <ul>
              {trades.map((t, i) => (
                <li key={i} onClick={() => setSelectedTrade(t)}>
                  {t.entry_date}: {t.root} {t.strike}{t.right} (PnL: {t.pnl.toFixed(2)})
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
