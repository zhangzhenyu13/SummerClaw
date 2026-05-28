/** Baseline vs Best Score bar chart — supports val + test grouped bars. */

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import type { EvalSplitResult } from '../api/types';

interface Props {
  baseline: number;
  best: number;
  valResult?: EvalSplitResult | null;
  testResult?: EvalSplitResult | null;
}

export const BaselineBarChart: React.FC<Props> = ({ baseline, best, valResult, testResult }) => {
  // Use new grouped format when val/test results are available
  const hasSplitResults = valResult || testResult;

  if (hasSplitResults) {
    const data: { name: string; Baseline: number; Best: number }[] = [];
    if (valResult) {
      data.push({
        name: 'Val',
        Baseline: valResult.score_no_skill,
        Best: valResult.score_with_skill,
      });
    }
    if (testResult) {
      data.push({
        name: 'Test',
        Baseline: testResult.score_no_skill,
        Best: testResult.score_with_skill,
      });
    }

    return (
      <div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis
              domain={[0, 'auto']}
              label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Legend />
            <Bar dataKey="Baseline" fill="#8c8c8c" radius={[6, 6, 0, 0]} barSize={50}>
              <LabelList
                dataKey="Baseline"
                position="top"
                formatter={(v: number) => v.toFixed(3)}
                style={{ fontSize: 11, fontWeight: 500 }}
              />
            </Bar>
            <Bar dataKey="Best" fill="#1677ff" radius={[6, 6, 0, 0]} barSize={50}>
              <LabelList
                dataKey="Best"
                position="top"
                formatter={(v: number) => v.toFixed(3)}
                style={{ fontSize: 11, fontWeight: 500 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        {data.map((d) => {
          const improvement = d.Baseline > 0 ? ((d.Best - d.Baseline) / d.Baseline) * 100 : 0;
          const delta = d.Best - d.Baseline;
          return (
            <div key={d.name} style={{ textAlign: 'center', marginTop: 6, color: improvement >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 500 }}>
              {d.name}: {improvement >= 0 ? '▲' : '▼'} {improvement >= 0 ? '+' : ''}{improvement.toFixed(2)}%
              &nbsp;|&nbsp; Δ = {delta >= 0 ? '+' : ''}{delta.toFixed(4)}
            </div>
          );
        })}
      </div>
    );
  }

  // Legacy single-bar format
  const data = [
    { name: 'Baseline', score: baseline, fill: '#8c8c8c' },
    { name: 'Best (Trained)', score: best, fill: '#1677ff' },
  ];
  const improvement = baseline > 0 ? ((best - baseline) / baseline) * 100 : 0;

  return (
    <div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis
            domain={[0, 'auto']}
            label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip formatter={(v: number) => v.toFixed(4)} />
          <Bar dataKey="score" radius={[6, 6, 0, 0]} barSize={80}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill={entry.fill} />
            ))}
            <LabelList
              dataKey="score"
              position="top"
              formatter={(v: number) => v.toFixed(4)}
              style={{ fontSize: 13, fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {baseline > 0 && best > 0 && (
        <div style={{ textAlign: 'center', marginTop: 8, color: improvement >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 500 }}>
          {improvement >= 0 ? '▲' : '▼'} Improvement: {improvement >= 0 ? '+' : ''}{improvement.toFixed(2)}%
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Δ = {(best - baseline).toFixed(4)}
        </div>
      )}
    </div>
  );
};
