/** Score chart — recharts LineChart wrapper. */

import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import type { ScorePoint } from '../api/types';

interface Props {
  data: ScorePoint[];
}

export const ScoreChart: React.FC<Props> = ({ data }) => {
  if (!data || data.length === 0) {
    return <div style={{ color: '#999', textAlign: 'center' }}>No data yet</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="step" label={{ value: 'Step', position: 'insideBottom', offset: -5 }} />
        <YAxis label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Line
          type="monotone"
          dataKey="score"
          stroke="#1677ff"
          strokeWidth={2}
          dot={{ r: 3 }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
