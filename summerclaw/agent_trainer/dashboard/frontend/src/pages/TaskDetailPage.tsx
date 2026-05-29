/** Task Detail page — Info, Logs, History, Deploy sub-tabs with real-time updates. */

import React, { useState, useEffect } from 'react';
import {
  Card, Tabs, Descriptions, Tag, Button, Space, Input, Table,
  message, Spin, Alert, List, Checkbox,
} from 'antd';
import {
  PlayCircleOutlined, StopOutlined, CopyOutlined,
  DownloadOutlined, ArrowLeftOutlined, DatabaseOutlined,
  ClockCircleOutlined, TrophyOutlined, AimOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useParams, useNavigate } from 'react-router-dom';
import {
  getTask, startTraining, cancelTraining, deploySkill,
  getTaskYaml, getTaskConfig, listTaskData, getTaskDataDownloadUrl,
  runEvalSingle, getEvalSingle,
} from '../api/client';
import { usePolling } from '../hooks/usePolling';
import type { TaskDetail, HistoryRow, EvalSingleResult } from '../api/types';
import { ScoreChart } from '../components/ScoreChart';
import { BaselineBarChart } from '../components/BaselineBarChart';
import { LogViewer } from '../components/LogViewer';
import { YamlConfigViewer } from '../components/YamlConfigViewer';

export const TaskDetailPage: React.FC = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [deployName, setDeployName] = useState('');
  const [skillPath, setSkillPath] = useState('');
  // Individual eval results: keyed by "val_no_skill", "val_with_skill", etc.
  const [evalResults, setEvalResults] = useState<Record<string, EvalSingleResult>>({});
  const [evalSelectedKeys, setEvalSelectedKeys] = useState<string[]>([
    'val_no_skill', 'val_with_skill', 'test_no_skill', 'test_with_skill',
  ]);
  const [evalRunning, setEvalRunning] = useState(false);

  // Real-time polling
  const { data: realtime } = usePolling(taskId || '', 5000);

  useEffect(() => {
    if (!taskId) return;
    (async () => {
      try {
        const d = await getTask(taskId);
        setDetail(d);
        // Build default deploy name
        const alg = d.algorithm || 'skill';
        let tname = d.task_id;
        if (tname.startsWith(`${alg}-`)) tname = tname.slice(alg.length + 1);
        setDeployName(`train-${alg}-${tname}`);
      } catch {
        message.error('Failed to load task detail');
      } finally {
        setLoading(false);
      }
    })();
  }, [taskId]);

  // Load existing eval results
  useEffect(() => {
    if (!taskId) return;
    (async () => {
      try {
        const res = await getEvalSingle(taskId);
        if (res.status === 'done' && res.results) {
          setEvalResults(res.results);
        }
      } catch {
        // no eval results yet
      }
    })();
  }, [taskId]);

  const handleStart = async () => {
    if (!taskId) return;
    try {
      const res = await startTraining(taskId, skillPath ? { skill_init_path: skillPath } : undefined);
      if ((res as Record<string, string>).error) {
        message.error((res as Record<string, string>).error);
      } else {
        message.success('Training started');
      }
    } catch {
      message.error('Failed to start training');
    }
  };

  const handleCancel = async () => {
    if (!taskId) return;
    try {
      await cancelTraining(taskId);
      message.info('Cancel requested');
    } catch {
      message.error('Failed to cancel');
    }
  };

  const handleCopyToCreate = () => {
    if (!taskId) return;
    navigate(`/create?copy_from=${encodeURIComponent(taskId)}`);
  };

  const handleDeploy = async () => {
    if (!taskId || !deployName) return;
    try {
      const res = await deploySkill(taskId, deployName) as Record<string, string>;
      if (res.error) {
        message.error(res.error);
      } else {
        message.success(`Deployed to ${res.path}`);
      }
    } catch {
      message.error('Deploy failed');
    }
  };

  const EVAL_OPTIONS = [
    { label: 'Val No Skill', value: 'val_no_skill' },
    { label: 'Val Best Skill', value: 'val_with_skill' },
    { label: 'Test No Skill', value: 'test_no_skill' },
    { label: 'Test Best Skill', value: 'test_with_skill' },
  ];

  const handleRunEvalSelected = async () => {
    if (!taskId || evalSelectedKeys.length === 0) return;
    setEvalRunning(true);
    message.loading({ content: `Evaluating ${evalSelectedKeys.length} item(s)…`, key: 'eval', duration: 0 });
    const newResults: Record<string, EvalSingleResult> = {};
    let hasError = false;
    for (const key of evalSelectedKeys) {
      const [split, mode] = key.startsWith('val_')
        ? ['val', key.slice(4)]
        : ['test', key.slice(5)];
      const withSkill = mode === 'with_skill';
      const label = `${split} ${withSkill ? 'Best Skill' : 'No Skill'}`;
      try {
        const res = await runEvalSingle(taskId, split as 'val' | 'test', withSkill) as Record<string, unknown>;
        if (res.error) {
          message.warning({ content: `${label}: ${res.error}`, key: 'eval' });
          hasError = true;
        } else if (res.result) {
          newResults[key] = res.result as EvalSingleResult;
        }
      } catch {
        message.warning({ content: `${label} evaluation failed`, key: 'eval' });
        hasError = true;
      }
    }
    // Merge new results into existing (update selected, keep others)
    setEvalResults(prev => ({ ...prev, ...newResults }));
    // Reload all results to get updated comparison data
    try {
      const fresh = await getEvalSingle(taskId);
      if (fresh.status === 'done' && fresh.results) {
        setEvalResults(fresh.results);
      }
    } catch {
      // use merged results
    }
    if (!hasError) {
      const scores = Object.entries(newResults).map(([k, r]) => `${k}: ${r.score.toFixed(4)}`).join(', ');
      message.success({ content: scores || 'Evaluation complete', key: 'eval' });
    }
    setEvalRunning(false);
  };

  // Show notification from realtime
  useEffect(() => {
    if (realtime?.notification) {
      message.info(realtime.notification, 8);
    }
  }, [realtime?.notification]);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  if (!detail) return <Alert type="error" message="Task not found" />;

  const isRunning = realtime?.is_running ?? false;
  const isStopping = realtime?.stop_requested ?? false;
  const isArchived = detail.archived ?? false;

  // Effective status: prefer realtime when engine is active
  const effectiveStatus = realtime?.status.status || detail.status;

  const STATUS_TAG: Record<string, { color: string; label: string }> = {
    running: { color: 'processing', label: 'Running' },
    completed: { color: 'success', label: 'Completed' },
    archived: { color: 'purple', label: 'Archived' },
    idle: { color: 'default', label: 'Idle' },
    failed: { color: 'error', label: 'Failed' },
    stopped: { color: 'warning', label: 'Stopped' },
    stopping: { color: 'warning', label: 'Stopping...' },
    queued: { color: 'cyan', label: 'Queued (waiting for workers budget)' },
  };
  const tagInfo = STATUS_TAG[effectiveStatus] || { color: 'default', label: effectiveStatus };

  // Start/Resume button logic:
  // - idle: can start fresh
  // - stopped: can resume from checkpoint
  // - queued: can force-start (bypasses scheduler budget)
  // - all others (running, completed, failed, archived): disabled
  const canStart = !isRunning && (effectiveStatus === 'idle' || effectiveStatus === 'stopped' || effectiveStatus === 'queued');
  const canRunEval = !isRunning && !evalRunning
    && ['completed', 'archived', 'stopped'].includes(effectiveStatus);
  const startLabel = effectiveStatus === 'stopped'
    ? 'Resume Training'
    : effectiveStatus === 'queued'
    ? 'Force Start (Bypass Queue)'
    : effectiveStatus === 'archived'
    ? 'Archived (Read-Only)'
    : effectiveStatus === 'completed'
    ? 'Completed (Read-Only)'
    : effectiveStatus === 'failed'
    ? 'Failed (Use Copy to Create)'
    : 'Start Training';

  // Format seconds to human-readable duration
  const fmtDuration = (s?: number | null): string => {
    if (s == null) return '—';
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    if (m < 60) return `${m}m ${sec}s`;
    const h = Math.floor(m / 60);
    const rem = m % 60;
    return `${h}h ${rem}m`;
  };
  // Format ISO timestamp to readable
  const fmtTs = (iso?: string | null): string => {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleString(); } catch { return iso; }
  };

  const historyColumns: ColumnsType<HistoryRow> = [
    { title: 'Step', dataIndex: 'step', key: 'step', width: 70 },
    { title: 'Epoch', dataIndex: 'epoch', key: 'epoch', width: 70 },
    {
      title: 'Score', dataIndex: 'score', key: 'score', width: 100,
      render: (v: number) => v.toFixed(4),
    },
    { title: 'Action', dataIndex: 'action', key: 'action' },
    { title: 'Hash', dataIndex: 'skill_hash', key: 'skill_hash', width: 100 },
    { title: 'Edits', dataIndex: 'edits_applied', key: 'edits_applied', width: 70 },
    { title: 'Rejected', dataIndex: 'edits_rejected', key: 'edits_rejected', width: 80 },
  ];

  const tabItems = [
    {
      key: 'info',
      label: 'Detail Info',
      children: (
        <div style={{ display: 'flex', gap: 24 }}>
          <div style={{ flex: 3 }}>
            <Descriptions bordered column={2} size="small">
              <Descriptions.Item label="Task ID">{detail.task_id}</Descriptions.Item>
              <Descriptions.Item label="Algorithm">{detail.algorithm}</Descriptions.Item>
              <Descriptions.Item label="Name">{detail.name || '—'}</Descriptions.Item>
              <Descriptions.Item label="Description">{detail.description || '—'}</Descriptions.Item>
              <Descriptions.Item label="Created">{detail.created}</Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={tagInfo.color}>{tagInfo.label}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label={<><AimOutlined /> Val No Skill</>}>
                {evalResults.val_no_skill?.score != null ? evalResults.val_no_skill.score.toFixed(4) : '—'}
              </Descriptions.Item>
              <Descriptions.Item label={<><TrophyOutlined /> Val Best Skill</>}>
                {evalResults.val_with_skill?.score != null ? evalResults.val_with_skill.score.toFixed(4) : '—'}
              </Descriptions.Item>
              <Descriptions.Item label={<><AimOutlined /> Test No Skill</>}>
                {evalResults.test_no_skill?.score != null ? evalResults.test_no_skill.score.toFixed(4) : '—'}
              </Descriptions.Item>
              <Descriptions.Item label={<><TrophyOutlined /> Test Best Skill</>}>
                {evalResults.test_with_skill?.score != null ? evalResults.test_with_skill.score.toFixed(4) : '—'}
              </Descriptions.Item>
              <Descriptions.Item label="Best Step">
                {realtime?.status.best_step ?? detail.best_step}
              </Descriptions.Item>
              <Descriptions.Item label="Total Steps">
                {realtime?.status.total_steps ?? detail.total_steps}
              </Descriptions.Item>
              <Descriptions.Item label="Total Epochs">
                {realtime?.status.total_epochs ?? detail.total_epochs}
              </Descriptions.Item>
              <Descriptions.Item label={<><ClockCircleOutlined /> Duration</>}>
                {fmtDuration(detail.duration_s)}
              </Descriptions.Item>
              <Descriptions.Item label="Started">{fmtTs(detail.started_at)}</Descriptions.Item>
              <Descriptions.Item label="Finished">{fmtTs(detail.finished_at)}</Descriptions.Item>
              <Descriptions.Item label="Path" span={2}>
                <code>{detail.path}</code>
              </Descriptions.Item>
            </Descriptions>

            {realtime?.data_status.loaded && (
              <Alert
                type="info" showIcon
                style={{ marginTop: 12 }}
                message={`Data loaded: ${Object.entries(realtime.data_status.splits || {}).map(([k, v]) => `${k}=${v}`).join(', ')}`}
                description={realtime.data_status.path ? `Path: ${realtime.data_status.path}` : undefined}
              />
            )}

            {isArchived && (
              <Alert
                type="warning" showIcon
                style={{ marginTop: 12 }}
                message="This task is archived (read-only)"
                description="Tasks can only run once. Use 'Copy to Create' to start a new training run with the same configuration."
              />
            )}

            {effectiveStatus === 'queued' && (
              <Alert
                type="info" showIcon
                style={{ marginTop: 12 }}
                message="Task is queued — waiting for workers budget"
                description="The scheduler will auto-start this task when enough workers budget becomes available. You can also force-start it using the button above."
              />
            )}

            <div style={{ marginTop: 16 }}>
              <Input
                placeholder="Initial Skill Path (optional, e.g. /path/to/skill.md)"
                value={skillPath}
                onChange={(e) => setSkillPath(e.target.value)}
                style={{ marginBottom: 8 }}
              />
            </div>
          </div>
          <div style={{ flex: 1, minWidth: 160 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary" block
                icon={<PlayCircleOutlined />}
                disabled={!canStart}
                onClick={handleStart}
              >
                {isRunning ? (isStopping ? 'Stopping...' : 'Running...') : startLabel}
              </Button>
              <Button
                danger block
                icon={<StopOutlined />}
                disabled={!isRunning || isStopping}
                onClick={handleCancel}
              >
                {isStopping ? 'Stopping...' : 'Stop Training'}
              </Button>
              <Button
                block icon={<CopyOutlined />}
                onClick={handleCopyToCreate}
              >
                Copy to Create
              </Button>
              <Card size="small" title="Eval" style={{ marginTop: 4 }}>
                <Checkbox.Group
                  options={EVAL_OPTIONS}
                  value={evalSelectedKeys}
                  onChange={(vals) => setEvalSelectedKeys(vals as string[])}
                  disabled={!canRunEval}
                  style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}
                />
                <Button
                  type="primary" size="small" block
                  disabled={!canRunEval || evalSelectedKeys.length === 0}
                  loading={evalRunning}
                  onClick={handleRunEvalSelected}
                >
                  Run Selected ({evalSelectedKeys.length})
                </Button>
              </Card>
            </Space>
          </div>
        </div>
      ),
    },
    {
      key: 'logs',
      label: 'Logs',
      children: (
        <LogViewer lines={realtime?.logs || []} />
      ),
    },
    {
      key: 'history',
      label: 'History',
      children: (
        <>
          <Table<HistoryRow>
            columns={historyColumns}
            dataSource={realtime?.history || []}
            rowKey={(r) => `${r.step}-${r.epoch}`}
            size="small"
            pagination={{ pageSize: 20, showSizeChanger: false }}
            scroll={{ y: 400 }}
          />
          {Object.keys(evalResults).length > 0 && (
            <Card title="Eval Results" style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', gap: 24 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <EvalResultsTable results={evalResults} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <BaselineBarChart results={evalResults} />
                </div>
              </div>
            </Card>
          )}
          {(realtime?.chart.length ?? 0) > 0 && (
            <Card title="Score Progress" style={{ marginTop: 16 }}>
              <ScoreChart data={realtime?.chart || []} />
            </Card>
          )}
        </>
      ),
    },
    {
      key: 'deploy',
      label: 'Deploy',
      children: (
        <Card>
          <p>Deploy the best skill to <code>~/.summerclaw/workspace/skills/</code></p>
          <Space>
            <Input
              addonBefore="Skill Name"
              value={deployName}
              onChange={(e) => setDeployName(e.target.value)}
              addonAfter=".md"
              style={{ width: 360 }}
            />
            <Button type="primary" onClick={handleDeploy}>
              Deploy Best Skill
            </Button>
          </Space>
        </Card>
      ),
    },
    {
      key: 'data',
      label: 'Data',
      children: (
        <DataDownloadTab taskId={taskId || ''} />
      ),
    },
    {
      key: 'yaml',
      label: 'YAML',
      children: (
        <YamlReadonlyTab taskId={taskId || ''} />
      ),
    },
  ];

  return (
    <div>
      <Button
        icon={<ArrowLeftOutlined />}
        style={{ marginBottom: 16 }}
        onClick={() => navigate('/')}
      >
        Back to Task List
      </Button>
      <Card title={`Task: ${detail.task_id}`}>
        <Tabs items={tabItems} defaultActiveKey="info" />
      </Card>
    </div>
  );
};

// -- Sub-components for tabs ------------------------------------------------

const DataDownloadTab: React.FC<{ taskId: string }> = ({ taskId }) => {
  const [splits, setSplits] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await listTaskData(taskId);
        setSplits(res.splits || {});
      } catch {
        // no data
      } finally {
        setLoading(false);
      }
    })();
  }, [taskId]);

  const splitNames = Object.keys(splits);

  if (loading) return <Spin />;

  if (!splitNames.length) {
    return (
      <Card title="Training Data">
        <Alert
          type="info" showIcon
          message="No data uploaded yet"
          description="Upload training data when creating a new task from the Create Task page."
        />
      </Card>
    );
  }

  return (
    <Card title="Training Data" extra={<Tag>{splitNames.length} splits</Tag>}>
      <List
        dataSource={splitNames}
        renderItem={(name) => (
          <List.Item
            actions={[
              <a
                key="download"
                href={getTaskDataDownloadUrl(taskId, name)}
                download={`${name}.json`}
              >
                <Button size="small" icon={<DownloadOutlined />}>
                  Download
                </Button>
              </a>,
            ]}
          >
            <List.Item.Meta
              avatar={<DatabaseOutlined style={{ fontSize: 20, color: '#1677ff' }} />}
              title={<strong>{name}</strong>}
              description={`${splits[name]} samples`}
            />
          </List.Item>
        )}
      />
    </Card>
  );
};

const YamlReadonlyTab: React.FC<{ taskId: string }> = ({ taskId }) => {
  const [yamlContent, setYamlContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        // Try /yaml endpoint first
        const res = await getTaskYaml(taskId);
        const r = res as Record<string, unknown>;
        if (r.error) {
          // Fallback: try /config endpoint (same source used by create page)
          try {
            const cfg = await getTaskConfig(taskId);
            if (cfg.yaml_content) {
              setYamlContent(cfg.yaml_content);
              return;
            }
          } catch {
            // ignore fallback failure
          }
          setError(String(r.error));
          return;
        }
        setYamlContent(res.content || '');
      } catch {
        setYamlContent('');
        setError('Failed to load YAML config');
      } finally {
        setLoading(false);
      }
    })();
  }, [taskId]);

  if (loading) return <Spin />;

  if (error && !yamlContent) {
    return (
      <Card title="YAML" size="small">
        <Alert type="warning" showIcon message={error} />
      </Card>
    );
  }

  return (
    <Card title="YAML" size="small">
      <YamlConfigViewer content={yamlContent} showRawToggle maxHeight={650} />
    </Card>
  );
};

// -- Eval results table -----------------------------------------------------

const EvalResultsTable: React.FC<{
  results: Record<string, EvalSingleResult>;
}> = ({ results }) => {
  // Build rows from results
  const rows = Object.entries(results).map(([key, r]) => ({
    key,
    split: r.split,
    mode: r.with_skill ? 'Best Skill' : 'No Skill',
    score: r.score,
    n_total: r.stats?.n_total ?? r.n_items,
    n_correct: r.stats?.n_correct ?? null,
    n_timeout: r.stats?.n_timeout ?? null,
    n_error: r.stats?.n_error ?? null,
    success_rate: r.stats?.success_rate ?? null,
    comparison: r.comparison,
  }));

  // Group by split for comparison display
  const splits = [...new Set(rows.map(r => r.split))];

  const columns = [
    { title: 'Split', dataIndex: 'split', key: 'split', width: 60 },
    { title: 'Mode', dataIndex: 'mode', key: 'mode', width: 80 },
    {
      title: 'Score',
      dataIndex: 'score',
      key: 'score',
      width: 70,
      render: (v: number) => v.toFixed(4),
    },
    {
      title: 'Total',
      dataIndex: 'n_total',
      key: 'n_total',
      width: 50,
    },
    {
      title: 'Correct',
      dataIndex: 'n_correct',
      key: 'n_correct',
      width: 60,
      render: (v: number | null) => v != null ? v : '—',
    },
    {
      title: 'Timeout',
      dataIndex: 'n_timeout',
      key: 'n_timeout',
      width: 60,
      render: (v: number | null) => {
        if (v == null) return '—';
        return <span style={{ color: v > 0 ? '#faad14' : undefined }}>{v}</span>;
      },
    },
    {
      title: 'Error',
      dataIndex: 'n_error',
      key: 'n_error',
      width: 50,
      render: (v: number | null) => {
        if (v == null) return '—';
        return <span style={{ color: v > 0 ? '#ff4d4f' : undefined }}>{v}</span>;
      },
    },
    {
      title: 'Rate',
      dataIndex: 'success_rate',
      key: 'success_rate',
      width: 60,
      render: (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : '—',
    },
  ];

  // Build comparison rows for splits that have both runs
  const comparisonRows = splits
    .map(split => {
      const noSkillResult = results[`${split}_no_skill`];
      const withSkillResult = results[`${split}_with_skill`];
      if (!noSkillResult || !withSkillResult || !withSkillResult.comparison) {
        return null;
      }
      const comp = withSkillResult.comparison;
      const allDelta = comp.all_items.with_skill_score - comp.all_items.no_skill_score;
      const completedDelta = comp.completed_items.with_skill_score - comp.completed_items.no_skill_score;
      return {
        key: split,
        split: split.charAt(0).toUpperCase() + split.slice(1),
        all_n: comp.all_items.n_total,
        all_no_skill: comp.all_items.no_skill_score,
        all_with_skill: comp.all_items.with_skill_score,
        all_delta: allDelta,
        completed_n: comp.completed_items.n_items,
        completed_no_skill: comp.completed_items.no_skill_score,
        completed_with_skill: comp.completed_items.with_skill_score,
        completed_delta: completedDelta,
        n_both_ok: comp.n_both_ok,
        n_at_least_one_ok: comp.n_at_least_one_ok,
        no_skill_rate: comp.no_skill_stats?.success_rate,
        with_skill_rate: comp.with_skill_stats?.success_rate,
      };
    })
    .filter(Boolean);

  const comparisonColumns = [
    { title: 'Split', dataIndex: 'split', key: 'split', width: 60 },
    {
      title: 'All Items',
      children: [
        { title: 'N', dataIndex: 'all_n', key: 'all_n', width: 40 },
        {
          title: 'No Skill',
          dataIndex: 'all_no_skill',
          key: 'all_no_skill',
          width: 70,
          render: (v: number) => v.toFixed(4),
        },
        {
          title: 'Skill',
          dataIndex: 'all_with_skill',
          key: 'all_with_skill',
          width: 70,
          render: (v: number) => v.toFixed(4),
        },
        {
          title: 'Delta',
          dataIndex: 'all_delta',
          key: 'all_delta',
          width: 70,
          render: (v: number) => (
            <span style={{ color: v >= 0 ? '#52c41a' : '#ff4d4f' }}>
              {v >= 0 ? '+' : ''}{v.toFixed(4)}
            </span>
          ),
        },
      ],
    },
    {
      title: 'Completed Items',
      children: [
        { title: 'N', dataIndex: 'completed_n', key: 'completed_n', width: 40 },
        {
          title: 'No Skill',
          dataIndex: 'completed_no_skill',
          key: 'completed_no_skill',
          width: 70,
          render: (v: number) => v.toFixed(4),
        },
        {
          title: 'Skill',
          dataIndex: 'completed_with_skill',
          key: 'completed_with_skill',
          width: 70,
          render: (v: number) => v.toFixed(4),
        },
        {
          title: 'Delta',
          dataIndex: 'completed_delta',
          key: 'completed_delta',
          width: 70,
          render: (v: number) => (
            <span style={{ color: v >= 0 ? '#52c41a' : '#ff4d4f' }}>
              {v >= 0 ? '+' : ''}{v.toFixed(4)}
            </span>
          ),
        },
      ],
    },
    {
      title: 'Both OK',
      dataIndex: 'n_both_ok',
      key: 'n_both_ok',
      width: 60,
    },
    {
      title: 'Success Rate',
      children: [
        {
          title: 'No Skill',
          dataIndex: 'no_skill_rate',
          key: 'no_skill_rate',
          width: 70,
          render: (v: number | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '—',
        },
        {
          title: 'Skill',
          dataIndex: 'with_skill_rate',
          key: 'with_skill_rate',
          width: 70,
          render: (v: number | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '—',
        },
      ],
    },
  ];

  return (
    <>
      <Table
        columns={columns}
        dataSource={rows}
        rowKey="key"
        size="small"
        pagination={false}
      />
      {comparisonRows.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <div style={{ marginBottom: 8, fontWeight: 500 }}>Comparison (Skill vs No Skill)</div>
          <Table
            columns={comparisonColumns}
            dataSource={comparisonRows}
            rowKey="key"
            size="small"
            pagination={false}
          />
        </div>
      )}
    </>
  );
};
