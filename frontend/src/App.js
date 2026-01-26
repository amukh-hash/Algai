import React from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import StrategyBuilder from './components/StrategyBuilder';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Institutional Algo Trading Platform</h1>
      </header>
      <main>
        <Dashboard />
        <StrategyBuilder />
      </main>
    </div>
  );
}

export default App;
