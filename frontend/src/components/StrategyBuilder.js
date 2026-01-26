import React, { useState, useCallback } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';

const initialNodes = [
  { id: '1', type: 'input', data: { label: 'Source: Binance' }, position: { x: 250, y: 5 } },
  { id: '2', data: { label: 'Model: PatchTST' }, position: { x: 250, y: 100 } },
  { id: '3', data: { label: 'Logic: Threshold > 0.5' }, position: { x: 250, y: 200 } },
  { id: '4', type: 'output', data: { label: 'Action: Buy' }, position: { x: 250, y: 300 } },
];

const initialEdges = [
  { id: 'e1-2', source: '1', target: '2', animated: true },
  { id: 'e2-3', source: '2', target: '3' },
  { id: 'e3-4', source: '3', target: '4' },
];

const StrategyBuilder = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  return (
    <div style={{ height: '500px', border: '1px solid #ddd', margin: '20px' }}>
      <h3>Strategy Builder (Visual Editor)</h3>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
};

export default StrategyBuilder;
