/** API client — thin wrapper around fetch for all dashboard endpoints. */

import type {
  TaskListResponse,
  TaskDetail,
  HistoryRow,
  ScorePoint,
  RealtimeSnapshot,
  CreateTaskParams,
  EvalTestResponse,
  SchedulerInfo,
  ToolCategory,
} from './types';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// -- Tasks ------------------------------------------------------------------

export async function listTasks(params: {
  search?: string;
  status?: string;
  sort?: string;
  asc?: boolean;
  page?: number;
  per_page?: number;
}): Promise<TaskListResponse> {
  const qs = new URLSearchParams();
  if (params.search) qs.set('search', params.search);
  if (params.status && params.status !== 'all') qs.set('status', params.status);
  if (params.sort) qs.set('sort', params.sort);
  if (params.asc !== undefined) qs.set('asc', String(params.asc));
  if (params.page) qs.set('page', String(params.page));
  if (params.per_page) qs.set('per_page', String(params.per_page));
  return request<TaskListResponse>(`/tasks?${qs}`);
}

export async function getTask(taskId: string): Promise<TaskDetail> {
  return request<TaskDetail>(`/tasks/${encodeURIComponent(taskId)}`);
}

export async function deleteTask(taskId: string): Promise<unknown> {
  return request(`/tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' });
}

export async function getTaskConfig(
  taskId: string,
): Promise<{ config: Record<string, unknown>; yaml_content: string; flat: Record<string, unknown>; task_id: string; has_skill: boolean; has_data: boolean }> {
  return request(`/tasks/${encodeURIComponent(taskId)}/config`);
}

export async function createTask(params: CreateTaskParams): Promise<unknown> {
  return request('/tasks', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// -- Training control -------------------------------------------------------

export async function startTraining(
  taskId: string,
  body?: { skill_init_path?: string },
): Promise<unknown> {
  return request(`/tasks/${encodeURIComponent(taskId)}/start`, {
    method: 'POST',
    body: JSON.stringify(body || {}),
  });
}

export async function cancelTraining(taskId: string): Promise<unknown> {
  return request(`/tasks/${encodeURIComponent(taskId)}/cancel`, { method: 'POST' });
}

// -- History & Skills -------------------------------------------------------

export async function getTaskHistory(
  taskId: string,
): Promise<{ history: HistoryRow[]; chart: ScorePoint[] }> {
  return request(`/tasks/${encodeURIComponent(taskId)}/history`);
}

export async function getTaskSkill(
  taskId: string,
  which: 'best' | 'current' = 'best',
): Promise<{ content: string; chars: number }> {
  return request(`/tasks/${encodeURIComponent(taskId)}/skill?which=${which}`);
}

// -- Deploy -----------------------------------------------------------------

export async function deploySkill(
  taskId: string,
  skillName: string,
): Promise<unknown> {
  return request(`/tasks/${encodeURIComponent(taskId)}/deploy`, {
    method: 'POST',
    body: JSON.stringify({ skill_name: skillName }),
  });
}

// -- YAML -------------------------------------------------------------------

export async function getTaskYaml(taskId: string): Promise<{ content: string; filename: string }> {
  return request(`/tasks/${encodeURIComponent(taskId)}/yaml`);
}

export async function uploadTaskYaml(
  taskId: string,
  content: string,
): Promise<unknown> {
  return request(`/tasks/${encodeURIComponent(taskId)}/yaml`, {
    method: 'POST',
    body: JSON.stringify({ content }),
  });
}

export async function getYamlTemplate(): Promise<{ content: string; filename: string }> {
  return request('/yaml/template');
}

// -- Data upload ------------------------------------------------------------

export async function uploadFile(
  taskId: string,
  formData: FormData,
): Promise<unknown> {
  formData.set('task_id', taskId);
  const res = await fetch(`${BASE}/upload/file`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  const data = await res.json() as Record<string, unknown>;
  if (data.error) throw new Error(String(data.error));
  return data;
}

export async function listTaskData(
  taskId: string,
): Promise<{ splits: Record<string, number>; task_id: string }> {
  return request(`/tasks/${encodeURIComponent(taskId)}/data`);
}

export function getTaskDataDownloadUrl(taskId: string, split: string): string {
  return `${BASE}/tasks/${encodeURIComponent(taskId)}/data/${encodeURIComponent(split)}`;
}

// -- Algorithms & Config ----------------------------------------------------

export async function listAlgorithms(): Promise<{ algorithms: string[] }> {
  return request('/algorithms');
}

export async function listMemoryAlgorithms(): Promise<{ algorithms: string[] }> {
  return request('/memory-algorithms');
}

export async function listAvailableTools(): Promise<{ categories: ToolCategory[] }> {
  return request('/tools');
}

export async function getConfig(): Promise<Record<string, unknown>> {
  return request('/config');
}

// -- Eval test --------------------------------------------------------------

export async function runEvalTest(taskId: string): Promise<EvalTestResponse | { error: string }> {
  return request(`/tasks/${encodeURIComponent(taskId)}/eval_test`, { method: 'POST' });
}

export async function getEvalTest(taskId: string): Promise<EvalTestResponse> {
  return request(`/tasks/${encodeURIComponent(taskId)}/eval_test`);
}

// -- Realtime snapshot ------------------------------------------------------

export async function getRealtimeSnapshot(taskId: string): Promise<RealtimeSnapshot> {
  return request(`/realtime?task_id=${encodeURIComponent(taskId)}`);
}

// -- Scheduler --------------------------------------------------------------

export async function getSchedulerInfo(): Promise<SchedulerInfo> {
  return request('/scheduler');
}
