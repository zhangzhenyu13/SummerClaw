/** Task List page — search, filter, sort, paginate, delete tasks. */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Card, Input, Select, Table, Tag, Button, Space, Popconfirm, message,
  Progress, Tooltip, Row, Col, Statistic,
} from 'antd';
import {
  SearchOutlined, ReloadOutlined, EyeOutlined, DeleteOutlined, CopyOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useNavigate } from 'react-router-dom';
import { listTasks, deleteTask, getSchedulerInfo } from '../api/client';
import type { Task, SchedulerInfo } from '../api/types';

const STATUS_COLORS: Record<string, string> = {
  running: 'processing',
  stopping: 'warning',
  completed: 'success',
  archived: 'purple',
  idle: 'default',
  failed: 'error',
  stopped: 'warning',
  queued: 'cyan',
};

export const TaskListPage: React.FC = () => {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortField, setSortField] = useState('created');
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [schedInfo, setSchedInfo] = useState<SchedulerInfo | null>(null);
  const perPage = 10;

  const fetchTasks = useCallback(async () => {
    setLoading(true);
    try {
      const res = await listTasks({
        search, status: statusFilter, sort: sortField,
        asc: sortAsc, page, per_page: perPage,
      });
      setTasks(res.tasks);
      setTotal(res.total);
    } catch (e) {
      message.error('Failed to load tasks');
    } finally {
      setLoading(false);
    }
  }, [search, statusFilter, sortField, sortAsc, page]);

  const fetchScheduler = useCallback(async () => {
    try {
      const info = await getSchedulerInfo();
      if (info.enabled) setSchedInfo(info);
    } catch {
      // scheduler may not be available
    }
  }, []);

  // Adaptive refresh: 3s when tasks are running/queued, otherwise 10s
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const refreshAll = useCallback(() => { fetchTasks(); fetchScheduler(); }, [fetchTasks, fetchScheduler]);

  useEffect(() => {
    refreshAll();
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [refreshAll]);

  // Re-schedule timer whenever schedInfo or tasks change
  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    const hasActive = tasks.some(t => t.status === 'running' || t.status === 'stopping' || t.status === 'queued')
      || (schedInfo && ((schedInfo.queued?.length ?? 0) > 0 || Object.keys(schedInfo.idle_pending ?? {}).length > 0));
    const interval = hasActive ? 3000 : 10000;
    timerRef.current = setInterval(refreshAll, interval);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [tasks, schedInfo, refreshAll]);

  const handleDelete = async (taskId: string) => {
    try {
      await deleteTask(taskId);
      message.success(`Task ${taskId} deleted`);
      fetchTasks();
    } catch {
      message.error('Delete failed');
    }
  };

  const columns: ColumnsType<Task> = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, r: Task) => (
        <a onClick={() => navigate(`/tasks/${encodeURIComponent(r.task_id)}`)}>
          <strong>{text || '(unnamed)'}</strong>
        </a>
      ),
    },
    {
      title: 'Task ID',
      dataIndex: 'task_id',
      key: 'task_id',
      render: (text: string) => (
        <code style={{ fontSize: 12, color: '#888' }}>{text}</code>
      ),
    },
    { title: 'Algorithm', dataIndex: 'algorithm', key: 'algorithm' },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (s: string) => <Tag color={STATUS_COLORS[s] || 'default'}>{s}</Tag>,
    },
    { title: 'Steps', dataIndex: 'total_steps', key: 'total_steps' },
    {
      title: 'Best Score',
      dataIndex: 'best_score',
      key: 'best_score',
      render: (v: number) => (v >= 0 ? v.toFixed(4) : '—'),
    },
    {
      title: 'Duration',
      dataIndex: 'duration_s',
      key: 'duration_s',
      render: (v: number | null | undefined) => {
        if (v == null) return '—';
        if (v < 60) return `${v.toFixed(1)}s`;
        const m = Math.floor(v / 60);
        const s = Math.round(v % 60);
        if (m < 60) return `${m}m ${s}s`;
        const h = Math.floor(m / 60);
        return `${h}h ${m % 60}m`;
      },
    },
    {
      title: 'Config',
      key: 'config',
      render: (_: unknown, r: Task) => `${r.epochs}e / bs=${r.batch_size}`,
    },
    {
      title: 'Workers',
      key: 'workers',
      dataIndex: 'effective_workers',
      width: 90,
      render: (ew: number | undefined, r: Task) => {
        const w = ew ?? r.workers ?? '?';
        const isRunning = r.status === 'running' || r.status === 'stopping';
        const isQueued = r.status === 'queued';
        return (
          <Tooltip title={`Config: w=${r.workers} → effective: ${w}`}>
            <Tag color={isRunning ? 'blue' : isQueued ? 'cyan' : 'default'}>
              {isRunning ? '⚡ ' : ''}{w}
            </Tag>
          </Tooltip>
        );
      },
    },
    { title: 'Created', dataIndex: 'created', key: 'created' },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, r: Task) => (
        <Space>
          <Button
            type="link" size="small" icon={<EyeOutlined />}
            onClick={() => navigate(`/tasks/${encodeURIComponent(r.task_id)}`)}
          >
            View
          </Button>
          <Button
            type="link" size="small" icon={<CopyOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              navigate(`/create?copy_from=${encodeURIComponent(r.task_id)}`);
            }}
          >
            Copy
          </Button>
          <Popconfirm
            title="Delete this task?"
            description={`This will permanently remove ${r.task_id}`}
            onConfirm={() => handleDelete(r.task_id)}
            okText="Delete"
            okType="danger"
          >
            <span onClick={(e) => e.stopPropagation()}>
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                Delete
              </Button>
            </span>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Card>
      {/* Scheduler Budget Bar */}
      {schedInfo && schedInfo.enabled && (
        <Card
          size="small"
          style={{ marginBottom: 16, background: '#fafafa' }}
          title={<><ThunderboltOutlined /> Workers Budget</>}
        >
          <Row gutter={24} align="middle">
            <Col span={12}>
              <Tooltip
                title={
                  <div>
                    <div>Used {schedInfo.used_workers} / {schedInfo.max_concurrency} total budget</div>
                    {Object.entries(schedInfo.running_tasks ?? {}).map(([tid, w]) => (
                      <div key={tid} style={{ fontSize: 12, opacity: 0.85 }}>⚡ {tid}: {w}w</div>
                    ))}
                  </div>
                }
              >
                <Progress
                  percent={Math.round((schedInfo.used_workers / Math.max(schedInfo.max_concurrency, 1)) * 100)}
                  format={() => `${schedInfo.used_workers} / ${schedInfo.max_concurrency}`}
                  strokeColor={schedInfo.left_budget === 0 ? '#ff4d4f' : '#1677ff'}
                />
              </Tooltip>
            </Col>
            <Col span={4}>
              <Statistic
                title="Left Budget"
                value={schedInfo.left_budget}
                suffix="workers"
                valueStyle={{ color: schedInfo.left_budget === 0 ? '#ff4d4f' : '#3f8600' }}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="Queued"
                value={(schedInfo.queued ?? []).length}
                suffix="tasks"
                valueStyle={{ color: (schedInfo.queued ?? []).length > 0 ? '#faad14' : '#8c8c8c' }}
              />
            </Col>
            <Col span={4}>
              <Statistic
                title="Idle Pending"
                value={Object.keys(schedInfo.idle_pending ?? {}).length}
                suffix="tasks"
              />
            </Col>
          </Row>
        </Card>
      )}

      <Space wrap style={{ marginBottom: 16 }}>
        <Input
          placeholder="Search tasks..."
          prefix={<SearchOutlined />}
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          style={{ width: 240 }}
          allowClear
        />
        <Select
          value={statusFilter}
          onChange={(v) => { setStatusFilter(v); setPage(1); }}
          style={{ width: 120 }}
          options={[
            { value: 'all', label: 'All Status' },
            { value: 'running', label: 'Running' },
            { value: 'queued', label: 'Queued' },
            { value: 'stopping', label: 'Stopping' },
            { value: 'completed', label: 'Completed' },
            { value: 'archived', label: 'Archived' },
            { value: 'stopped', label: 'Stopped' },
            { value: 'failed', label: 'Failed' },
            { value: 'idle', label: 'Idle' },
          ]}
        />
        <Select
          value={sortField}
          onChange={(v) => { setSortField(v); setPage(1); }}
          style={{ width: 140 }}
          options={[
            { value: 'created', label: 'Sort: Created' },
            { value: 'best_score', label: 'Sort: Score' },
            { value: 'total_steps', label: 'Sort: Steps' },
            { value: 'algorithm', label: 'Sort: Algorithm' },
          ]}
        />
        <Select
          value={sortAsc ? 'asc' : 'desc'}
          onChange={(v) => setSortAsc(v === 'asc')}
          style={{ width: 80 }}
          options={[
            { value: 'desc', label: 'Desc' },
            { value: 'asc', label: 'Asc' },
          ]}
        />
        <Button icon={<ReloadOutlined />} onClick={refreshAll}>
          Refresh
        </Button>
      </Space>

      <Table<Task>
        columns={columns}
        dataSource={tasks}
        rowKey="task_id"
        loading={loading}
        pagination={{
          current: page,
          pageSize: perPage,
          total,
          showTotal: (t) => `Total ${t} tasks`,
          onChange: (p) => setPage(p),
          showSizeChanger: false,
        }}
        size="middle"
        onRow={(record) => ({
          onClick: () => navigate(`/tasks/${encodeURIComponent(record.task_id)}`),
          style: { cursor: 'pointer' },
        })}
      />
    </Card>
  );
};
