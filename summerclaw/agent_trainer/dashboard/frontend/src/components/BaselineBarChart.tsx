/** Baseline vs Best Score bar chart — shows all items and completed items comparison. */

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList,
} from 'recharts';
import type { EvalSingleResult } from '../api/types';

interface Props {
  results: Record<string, EvalSingleResult>;
}

export const BaselineBarChart: React.FC<Props> = ({ results }) => {
  // Build chart data from comparison data
  const splits = [...new Set(Object.values(results).map(r => r.split))];

  const data: {
    name: string;
    'All - No Skill': number;
    'All - Skill': number;
    'Completed - No Skill': number;
    'Completed - Skill': number;
  }[] = [];

  for (const split of splits) {
    const withSkillResult = results[`${split}_with_skill`];
    const noSkillResult = results[`${split}_no_skill`];

    if (!withSkillResult || !noSkillResult) continue;

    const comp = withSkillResult.comparison;
    if (!comp) continue;

    data.push({
      name: split.charAt(0).toUpperCase() + split.slice(1),
      'All - No Skill': comp.all_items.no_skill_score,
      'All - Skill': comp.all_items.with_skill_score,
      'Completed - No Skill': comp.completed_items.no_skill_score,
      'Completed - Skill': comp.completed_items.with_skill_score,
    });
  }

  if (!data.length) return null;

  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 20, right: 20, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis
            domain={[0, 'auto']}
            label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip formatter={(v: number) => v.toFixed(4)} />
          <Legend />
          <Bar dataKey="All - No Skill" fill="#bfbfbf" radius={[4, 4, 0, 0]} barSize={30}>
            <LabelList dataKey="All - No Skill" position="top" formatter={(v: number) => v.toFixed(2)} style={{ fontSize: 10 }} />
          </Bar>
          <Bar dataKey="All - Skill" fill="#8c8c8c" radius={[4, 4, 0, 0]} barSize={30}>
            <LabelList dataKey="All - Skill" position="top" formatter={(v: number) => v.toFixed(2)} style={{ fontSize: 10 }} />
          </Bar>
          <Bar dataKey="Completed - No Skill" fill="#91d5ff" radius={[4, 4, 0, 0]} barSize={30}>
            <LabelList dataKey="Completed - No Skill" position="top" formatter={(v: number) => v.toFixed(2)} style={{ fontSize: 10 }} />
          </Bar>
          <Bar dataKey="Completed - Skill" fill="#1677ff" radius={[4, 4, 0, 0]} barSize={30}>
            <LabelList dataKey="Completed - Skill" position="top" formatter={(v: number) => v.toFixed(2)} style={{ fontSize: 10 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {data.map((d) => {
        const allDelta = d['All - Skill'] - d['All - No Skill'];
        const completedDelta = d['Completed - Skill'] - d['Completed - No Skill'];
        return (
          <div key={d.name} style={{ textAlign: 'center', marginTop: 8 }}>
            <div style={{ color: allDelta >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 500 }}>
              {d.name} (All): Δ = {allDelta >= 0 ? '+' : ''}{allDelta.toFixed(4)}
            </div>
            <div style={{ color: completedDelta >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 500 }}>
              {d.name} (Completed): Δ = {completedDelta >= 0 ? '+' : ''}{completedDelta.toFixed(4)}
            </div>
          </div>
        );
      })}
    </div>
  );
};
