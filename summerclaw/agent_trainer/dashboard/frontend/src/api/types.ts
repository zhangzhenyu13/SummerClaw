/** Shared TypeScript types for the dashboard API. */

export interface Task {
  task_id: string;
  name: string;
  description: string;
  algorithm: string;
  status: 'running' | 'stopping' | 'completed' | 'archived' | 'idle' | 'failed' | 'stopped' | 'queued';
  archived: boolean;
  best_score: number;
  baseline_score: number;
  best_step: number;
  total_steps: number;
  total_epochs: number;
  epochs: number | string;
  batch_size: number | string;
  workers: number | string;
  effective_workers: number;
  created: string;
  path: string;
  notes: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration_s?: number | null;
}

export interface TaskListResponse {
  tasks: Task[];
  page: number;
  total_pages: number;
  total: number;
  start: number;
  end: number;
}

export interface TaskDetail {
  task_id: string;
  name: string;
  description: string;
  algorithm: string;
  created: string;
  status: string;
  archived: boolean;
  best_score: number;
  baseline_score: number;
  best_step: number;
  total_steps: number;
  total_epochs: number;
  path: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration_s?: number | null;
}

export interface HistoryRow {
  step: number;
  epoch: number;
  score: number;
  action: string;
  skill_hash: string;
  edits_applied: number;
  edits_rejected: number;
}

export interface ScorePoint {
  step: number;
  score: number;
}

export interface StatusInfo {
  status: string;
  best_score: number;
  baseline_score: number;
  best_step: number;
  total_steps: number;
  total_epochs: number;
}

export interface DataStatus {
  loaded: boolean;
  splits?: Record<string, number>;
  path?: string;
}

export interface RealtimeSnapshot {
  status: StatusInfo;
  history: HistoryRow[];
  chart: ScorePoint[];
  logs: string[];
  data_status: DataStatus;
  is_running: boolean;
  stop_requested: boolean;
  notification: string | null;
  deploy_name: string;
}

export interface EvalSplitResult {
  n_items: number;
  score_with_skill: number;
  score_no_skill: number;
  delta: number;
  improvement_pct: number | null;
}

export interface EvalTestSummary {
  val: EvalSplitResult | null;
  test: EvalSplitResult | null;
  best_skill_chars: number;
}

export interface EvalTestResponse {
  status: 'done' | 'not_found' | 'running';
  summary?: EvalTestSummary;
}

export interface CreateTaskParams {
  name: string;
  description?: string;
  algorithm: string;
  skill_path?: string;
  copy_from?: string;
  epochs: number;
  batch_size: number;
  workers?: number;
  seed: number;
  learning_rate: number;
  lr_scheduler: string;
  update_mode: string;
  slow_update: boolean;
  meta_skill: boolean;
  reasoning_effort: string;
  yaml_content?: string;
  memory_algorithm?: string | null;
  enabled_tools?: string[];
}

export interface SchedulerInfo {
  enabled: boolean;
  max_concurrency: number;
  used_workers: number;
  left_budget: number;
  idle_pending: Record<string, number>;
  queued: string[];
  running_tasks: Record<string, number>;
}

export interface ToolCategory {
  key: string;
  label: string;
  default_excluded: boolean;
  tools: string[];
}
