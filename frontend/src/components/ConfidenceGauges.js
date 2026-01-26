import React from 'react';
import { PieChart, Pie, Cell, Tooltip } from 'recharts';

const ConfidenceGauges = ({ direction, volatility }) => {
  const dataDir = [
    { name: 'Up', value: direction },
    { name: 'Down', value: 1 - direction },
  ];
  const COLORS = ['#00C49F', '#FF8042'];

  return (
    <div className="card">
      <h3>AI Confidence</h3>
      <div style={{ display: 'flex', justifyContent: 'space-around' }}>
        <div>
          <h4>Direction</h4>
          <PieChart width={100} height={100}>
            <Pie data={dataDir} innerRadius={30} outerRadius={40} fill="#8884d8" paddingAngle={5} dataKey="value">
              {dataDir.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
          <p>{(direction * 100).toFixed(1)}% UP</p>
        </div>
        <div>
          <h4>Volatility</h4>
          <h2>{volatility.toFixed(3)}</h2>
          <p>Sigma</p>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceGauges;
