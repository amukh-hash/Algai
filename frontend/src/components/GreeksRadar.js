import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

const GreeksRadar = ({ greeks }) => {
  if (!greeks) return null;
  
  const data = [
    { subject: 'Delta', A: Math.abs(greeks.delta), fullMark: 1 },
    { subject: 'Gamma', A: greeks.gamma * 10, fullMark: 1 }, // Scale for visibility
    { subject: 'Theta', A: Math.abs(greeks.theta), fullMark: 1 },
    { subject: 'Vega', A: greeks.vega, fullMark: 1 },
  ];

  return (
    <div className="card">
      <h3>Greeks Personality</h3>
      <div style={{ width: '100%', height: 200 }}>
        <ResponsiveContainer>
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" />
            <PolarRadiusAxis angle={30} domain={[0, 1]} />
            <Radar name="Trade" dataKey="A" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default GreeksRadar;
