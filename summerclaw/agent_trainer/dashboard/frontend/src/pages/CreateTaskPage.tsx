/** Create Task page — form for creating new training tasks.
 *  Supports ?copy_from=<taskId> to pre-fill from an existing task's config.
 *  YAML config is editable directly; data upload + split config included.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Card, Form, Input, InputNumber, Select, Switch, Button, Space, Tabs, Upload,
  message, Alert, Spin, Divider,
} from 'antd';
import { CloudUploadOutlined, DeleteOutlined } from '@ant-design/icons';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { createTask, listAlgorithms, getYamlTemplate, getTaskConfig, uploadFile, listMemoryAlgorithms, listAvailableTools } from '../api/client';
import type { ToolCategory } from '../api/types';
import { YamlConfigViewer, parseStructuredYaml } from '../components/YamlConfigViewer';

export const CreateTaskPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const copyFrom = searchParams.get('copy_from') || '';
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);
  const [algorithms, setAlgorithms] = useState<string[]>(['skillopt']);
  const [memoryAlgorithms, setMemoryAlgorithms] = useState<string[]>(['naive_memory']);
  const [toolCategories, setToolCategories] = useState<ToolCategory[]>([]);
  const [toolsLoading, setToolsLoading] = useState(true);
  const [copyLoading, setCopyLoading] = useState(false);
  const [copySource, setCopySource] = useState<string>('');
  const [hasSourceSkill, setHasSourceSkill] = useState(false);
  const [hasSourceData, setHasSourceData] = useState(false);
  const [yamlContent, setYamlContent] = useState<string>('');
  const [mainFile, setMainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [scorerFile, setScorerFile] = useState<File | null>(null);
  const scorerMode = Form.useWatch('scorer_mode', form);
  // ---------- enabled_tools managed by local state (NOT through Form) ----------
  // Ant Design Checkbox.Group is completely non-interactive in this environment,
  // so we use plain HTML checkboxes driven by local useState.
  const [enabledTools, setEnabledTools] = useState<string[]>([]);
  const initialized = useRef(false);
  const yamlSynced = useRef(false);

  // ---------------------------------------------------------------------------
  // Parse YAML content → extract form field values
  // ---------------------------------------------------------------------------
  const syncFormFromYaml = (yaml: string) => {
    if (!yaml || yamlSynced.current) return;
    yamlSynced.current = true;
    const sections = parseStructuredYaml(yaml);
    const vals: Record<string, unknown> = {};
    for (const sec of sections) {
      for (const p of sec.params) {
        switch (p.key) {
          case 'num_epochs':  vals.epochs       = Number(p.value) || 3; break;
          case 'batch_size':  vals.batch_size   = Number(p.value) || 5; break;
          case 'workers':     vals.workers       = Number(p.value) || 0; break;
          case 'accumulation': vals.accumulation = Number(p.value) || 1; break;
          case 'seed':        vals.seed          = Number(p.value) || 42; break;
          case 'learning_rate': vals.learning_rate = Number(p.value) || 4; break;
          case 'min_learning_rate': vals.min_learning_rate = Number(p.value) || 2; break;
          case 'lr_scheduler': vals.lr_scheduler = p.value || 'constant'; break;
          case 'skill_update_mode': vals.update_mode = p.value || 'patch'; break;
          case 'use_slow_update': vals.slow_update = p.value === 'true'; break;
          case 'slow_update_samples': vals.slow_update_samples = Number(p.value) || 20; break;
          case 'use_meta_skill':  vals.meta_skill  = p.value === 'true'; break;
          case 'longitudinal_pair_policy': vals.longitudinal_pair_policy = p.value || 'mixed'; break;
          case 'minibatch_size': vals.minibatch_size = Number(p.value) || 8; break;
          case 'merge_batch_size': vals.merge_batch_size = Number(p.value) || 8; break;
          case 'max_analyst_rounds': vals.max_analyst_rounds = Number(p.value) || 3; break;
          case 'failure_only': vals.failure_only = p.value === 'true'; break;
          case 'reasoning_effort': vals.reasoning_effort = p.value || 'medium'; break;
          case 'rewrite_reasoning_effort': vals.rewrite_reasoning_effort = p.value || ''; break;
          case 'rewrite_max_completion_tokens': vals.rewrite_max_completion_tokens = Number(p.value) || 64000; break;
          case 'sel_env_num': vals.sel_env_num = Number(p.value) || 0; break;
          case 'test_env_num': vals.test_env_num = Number(p.value) || 0; break;
          case 'eval_test': vals.eval_test = p.value === 'true'; break;
          case 'split_seed':  vals.split_seed   = Number(p.value) || 42; break;
          case 'exec_timeout': vals.exec_timeout = Number(p.value) || 120; break;
          case 'memory_algorithm': vals.memory_algorithm = p.value || 'null'; break;
          case 'enabled_tools':
            try {
              const raw = String(p.value).trim();
              if (raw === '[]' || raw === '') vals.enabled_tools = [];
              else if (raw.startsWith('['))
                vals.enabled_tools = raw.replace(/^\[|\]$/g, '').split(',').map((s) => {
                  const t = s.trim();
                  return (t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))
                    ? t.slice(1, -1) : t;
                }).filter(Boolean);
              else vals.enabled_tools = [];
            } catch { vals.enabled_tools = []; }
            break;
        }
      }
    }
    if (Object.keys(vals).length > 0) {
      // enabled_tools is managed by local state, not Form
      const toolsFromYaml = vals.enabled_tools as string[] | undefined;
      delete vals.enabled_tools;
      if (toolsFromYaml) setEnabledTools(toolsFromYaml);
      if (Object.keys(vals).length > 0) form.setFieldsValue(vals);
    }
  };

  useEffect(() => {
    listAlgorithms().then((res) => setAlgorithms(res.algorithms)).catch(() => {});
    listMemoryAlgorithms().then((res) => setMemoryAlgorithms(res.algorithms)).catch(() => {});
    listAvailableTools().then((res) => {
      const cats = res.categories || [];
      setToolCategories(cats);
      // Default: select only non-excluded categories (only when not in copy-from mode and not yet initialized)
      if (!copyFrom && !initialized.current) {
        const defaultTools = cats.filter((c) => !c.default_excluded).flatMap((c) => c.tools);
        setEnabledTools(defaultTools);
      }
    }).catch(() => {}).finally(() => setToolsLoading(false));
    // Load default YAML template (non-copy mode)
    if (!copyFrom) {
      getYamlTemplate()
        .then((res) => {
          setYamlContent(res.content);
          // Sync form fields from YAML template values
          syncFormFromYaml(res.content);
        })
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Pre-fill form when copy_from is provided
  useEffect(() => {
    if (!copyFrom || initialized.current) return;
    initialized.current = true;
    setCopyLoading(true);
    getTaskConfig(copyFrom)
      .then((res) => {
        const r = res as Record<string, unknown>;
        if (r.error) {
          message.error(String(r.error));
          return;
        }
        const cfg = res.config;
        const flat = res.flat;
        setCopySource(copyFrom);
        setHasSourceSkill(Boolean(res.has_skill));
        setHasSourceData(Boolean(res.has_data));
        form.setFieldsValue({
          description: (cfg.description as string) || '',
          algorithm: (cfg.algorithm as string) || 'skillopt',
          skill_path: res.has_skill
            ? `(copied from ${copyFrom})`
            : ((cfg.skill_init as string) || ''),
          epochs: Number(flat.num_epochs ?? cfg.num_epochs ?? 3),
          batch_size: Number(flat.batch_size ?? cfg.batch_size ?? 5),
          workers: Number(flat.workers ?? cfg.workers ?? 0),
          accumulation: Number(flat.accumulation ?? cfg.accumulation ?? 1),
          seed: Number(flat.seed ?? cfg.seed ?? 42),
          learning_rate: Number(flat.edit_budget ?? flat.learning_rate ?? 4),
          min_learning_rate: Number(flat.min_edit_budget ?? flat.min_learning_rate ?? 2),
          lr_scheduler: (flat.lr_scheduler as string) || 'constant',
          update_mode: (flat.skill_update_mode as string) || (flat.update_mode as string) || 'patch',
          slow_update: Boolean(flat.use_slow_update ?? true),
          slow_update_samples: Number(flat.slow_update_samples ?? cfg.slow_update_samples ?? 20),
          meta_skill: Boolean(flat.use_meta_skill ?? true),
          longitudinal_pair_policy: (flat.longitudinal_pair_policy as string) || 'mixed',
          minibatch_size: Number(flat.minibatch_size ?? cfg.minibatch_size ?? 8),
          merge_batch_size: Number(flat.merge_batch_size ?? cfg.merge_batch_size ?? 8),
          max_analyst_rounds: Number(flat.max_analyst_rounds ?? cfg.max_analyst_rounds ?? 3),
          failure_only: Boolean(flat.failure_only ?? false),
          reasoning_effort: (flat.reasoning_effort as string) || 'medium',
          rewrite_reasoning_effort: (flat.rewrite_reasoning_effort as string) || '',
          rewrite_max_completion_tokens: Number(flat.rewrite_max_completion_tokens ?? cfg.rewrite_max_completion_tokens ?? 64000),
          sel_env_num: Number(flat.sel_env_num ?? cfg.sel_env_num ?? 0),
          test_env_num: Number(flat.test_env_num ?? cfg.test_env_num ?? 0),
          eval_test: Boolean(flat.eval_test ?? true),
          exec_timeout: Number(flat.exec_timeout ?? cfg.exec_timeout ?? 120),
          memory_algorithm: flat.memory_algorithm != null ? String(flat.memory_algorithm) : 'null',
          enabled_tools: Array.isArray(flat.enabled_tools) ? flat.enabled_tools as string[] : [],
        });
        // Sync local tools state from copy-from config
        if (Array.isArray(flat.enabled_tools)) {
          setEnabledTools(flat.enabled_tools as string[]);
        }
        if (res.yaml_content) {
          setYamlContent(res.yaml_content);
          // Sync form fields from copied task's YAML
          syncFormFromYaml(res.yaml_content);
        }
        message.info(`Pre-filled from task: ${copyFrom}`);
      })
      .catch((e) => {
        message.error(`Failed to load config: ${e instanceof Error ? e.message : 'unknown'}`);
      })
      .finally(() => setCopyLoading(false));
  }, [copyFrom, form]);

  const handleSubmit = async (values: Record<string, unknown>) => {
    setSubmitting(true);
    try {
      // Ensure YAML is in sync with the latest form values before submission
      const syncedYaml = patchYamlWithFormValues(yamlContent, values);
      const res = await createTask({
        name: values.name as string,
        description: (values.description as string) || '',
        algorithm: values.algorithm as string,
        skill_path: copyFrom ? '' : ((values.skill_path as string) || ''),
        copy_from: copyFrom || undefined,
        epochs: (values.epochs as number) || 3,
        batch_size: (values.batch_size as number) || 5,
        workers: (values.workers as number) ?? 0,
        seed: (values.seed as number) || 42,
        learning_rate: (values.learning_rate as number) || 4,
        lr_scheduler: values.lr_scheduler as string || 'constant',
        update_mode: values.update_mode as string || 'patch',
        slow_update: values.slow_update as boolean ?? true,
        meta_skill: values.meta_skill as boolean ?? true,
        reasoning_effort: values.reasoning_effort as string || 'medium',
        yaml_content: syncedYaml || '',
        memory_algorithm: (values.memory_algorithm as string) || 'null',
        enabled_tools: enabledTools,
      });
      const r = res as Record<string, string>;
      if (r.error) {
        message.error(r.error);
        return;
      }
      const createdId = r.task_id || '';
      if (r.copied_skill_path) {
        message.success(`Initial skill copied from ${copyFrom} → ${r.copied_skill_path}`);
      }

      // Upload data files if provided
      // When copy-from has data AND user didn't upload a new file, skip upload (data already copied by backend)
      // When user uploads a new file, it will override the copied data
      if (mainFile && createdId) {
        try {
          const formData = new FormData();
          // Convert RcFile to proper File object for FormData compatibility
          const mainFileObj = new File([mainFile], mainFile.name, { type: mainFile.type });
          formData.append('main_file', mainFileObj);
          if (testFile) {
            const testFileObj = new File([testFile], testFile.name, { type: testFile.type });
            formData.append('test_file', testFileObj);
          }
          formData.append('train_ratio', String(values.train_ratio ?? 7));
          formData.append('val_ratio', String(values.val_ratio ?? 2));
          formData.append('test_ratio', String(values.test_ratio ?? 1));
          formData.append('seed', String(values.split_seed ?? values.seed ?? 42));
          formData.append('scorer_mode', String(values.scorer_mode ?? 'exact_match'));
          if (scorerFile) {
            const scorerFileObj = new File([scorerFile], scorerFile.name, { type: scorerFile.type });
            formData.append('scorer_file', scorerFileObj);
          }
          await uploadFile(createdId, formData);
          message.success('Data uploaded and split');
        } catch (e) {
          message.warning(`Task created but data upload failed: ${e instanceof Error ? e.message : 'unknown'}`);
        }
      }

      if (r.data_copied) {
        message.success(`Data copied from ${copyFrom}`);
      }
      message.success(`Task ${createdId} created`);
      if (createdId) navigate(`/tasks/${encodeURIComponent(createdId)}`);
    } catch (e) {
      message.error(`Failed: ${e instanceof Error ? e.message : 'unknown'}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = () => {
    form.resetFields();
    setMainFile(null);
    setTestFile(null);
    setScorerFile(null);
  };

  const hasTestFile = !!testFile;

  // ---------------------------------------------------------------------------
  // Sync YAML editor content when form values change
  // ---------------------------------------------------------------------------
  const patchYamlWithFormValues = (yaml: string, values: Record<string, unknown>): string => {
    if (!yaml) return yaml;
    const lines = yaml.split('\n');
    const mappings: { yamlKey: string; formField: string; type: 'num' | 'str' | 'bool' | 'list'; section: string }[] = [
      { yamlKey: 'num_epochs',      formField: 'epochs',           type: 'num',  section: 'train' },
      { yamlKey: 'batch_size',      formField: 'batch_size',       type: 'num',  section: 'train' },
      { yamlKey: 'workers',         formField: 'workers',          type: 'num',  section: 'train' },
      { yamlKey: 'accumulation',    formField: 'accumulation',     type: 'num',  section: 'train' },
      { yamlKey: 'seed',            formField: 'seed',             type: 'num',  section: 'train' },
      { yamlKey: 'learning_rate',   formField: 'learning_rate',    type: 'num',  section: 'optimizer' },
      { yamlKey: 'min_learning_rate', formField: 'min_learning_rate', type: 'num', section: 'optimizer' },
      { yamlKey: 'lr_scheduler',    formField: 'lr_scheduler',     type: 'str',  section: 'optimizer' },
      { yamlKey: 'skill_update_mode', formField: 'update_mode',    type: 'str',  section: 'optimizer' },
      { yamlKey: 'use_slow_update', formField: 'slow_update',      type: 'bool', section: 'optimizer' },
      { yamlKey: 'slow_update_samples', formField: 'slow_update_samples', type: 'num', section: 'optimizer' },
      { yamlKey: 'use_meta_skill',  formField: 'meta_skill',       type: 'bool', section: 'optimizer' },
      { yamlKey: 'longitudinal_pair_policy', formField: 'longitudinal_pair_policy', type: 'str', section: 'optimizer' },
      { yamlKey: 'minibatch_size',  formField: 'minibatch_size',   type: 'num',  section: 'gradient' },
      { yamlKey: 'merge_batch_size', formField: 'merge_batch_size', type: 'num', section: 'gradient' },
      { yamlKey: 'max_analyst_rounds', formField: 'max_analyst_rounds', type: 'num', section: 'gradient' },
      { yamlKey: 'failure_only',    formField: 'failure_only',     type: 'bool', section: 'gradient' },
      { yamlKey: 'reasoning_effort', formField: 'reasoning_effort', type: 'str', section: 'model' },
      { yamlKey: 'rewrite_reasoning_effort', formField: 'rewrite_reasoning_effort', type: 'str', section: 'model' },
      { yamlKey: 'rewrite_max_completion_tokens', formField: 'rewrite_max_completion_tokens', type: 'num', section: 'model' },
      { yamlKey: 'sel_env_num',     formField: 'sel_env_num',      type: 'num',  section: 'evaluation' },
      { yamlKey: 'test_env_num',    formField: 'test_env_num',     type: 'num',  section: 'evaluation' },
      { yamlKey: 'eval_test',       formField: 'eval_test',        type: 'bool', section: 'evaluation' },
      { yamlKey: 'exec_timeout',    formField: 'exec_timeout',     type: 'num',  section: 'env' },
      { yamlKey: 'memory_algorithm', formField: 'memory_algorithm', type: 'str', section: 'env' },
      { yamlKey: 'enabled_tools',   formField: 'enabled_tools',    type: 'list', section: 'env' },
    ];
    for (const m of mappings) {
      if (!(m.formField in values)) continue;
      const val = values[m.formField];
      let replacement: string;
      if (m.type === 'num')    replacement = String(Number(val) || 0);
      else if (m.type === 'bool') replacement = String(Boolean(val));
      else if (m.type === 'list') {
        const arr = Array.isArray(val) ? val : [];
        replacement = arr.length > 0
          ? '[' + arr.map((s: string) => `"${String(s).replace(/"/g, '\\"')}"`).join(', ') + ']'
          : '[]';
      }
      else                       replacement = String(val);
      const re = new RegExp(`^(\\s*${m.yamlKey}\\s*:\\s*).*$`);
      let found = false;
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line !== undefined && re.test(line)) {
          // Normalize the prefix to always end with ": " (the regex capture
          // group may lack a trailing space when the key has no inline value,
          // e.g. "enabled_tools:" — producing "enabled_tools:[...]" which is
          // invalid YAML).
          const prefix = (line.match(re)?.[1] ?? '').replace(/\s+$/, '') + ' ';
          lines[i] = prefix + replacement;
          // For list types, remove any trailing block-style list items
          // that yaml.dump may have written (e.g. "  - item1\n  - item2").
          // Skip blank lines between items — they don't end the block list.
          if (m.type === 'list') {
            let j = i + 1;
            while (j < lines.length && lines[j] !== undefined) {
              const next = lines[j] ?? '';
              if (/^\s+-\s/.test(next)) { j++; continue; }
              if (next.trim() === '') { j++; continue; }
              break;
            }
            if (j > i + 1) {
              lines.splice(i + 1, j - i - 1);
            }
          }
          found = true;
          break;
        }
      }
      // If key not found in YAML, append under the correct section
      if (!found) {
        const sectionHeaderRe = new RegExp(`^${m.section}\\s*:\\s*$`);
        for (let i = 0; i < lines.length; i++) {
          if (sectionHeaderRe.test(lines[i] ?? '')) {
            // Find the end of this section (next section header or EOF)
            let insertAt = i + 1;
            while (insertAt < lines.length) {
              const next = lines[insertAt];
              if (next !== undefined && /^[^\s#]/.test(next) && next.trim().endsWith(':') && !next.startsWith(' ') && !next.startsWith('\t')) {
                break;
              }
              insertAt++;
            }
            lines.splice(insertAt, 0, `  ${m.yamlKey}: ${replacement}`);
            break;
          }
        }
      }
    }
    return lines.join('\n');
  };

  // Toggle a single tool on/off
  const toggleTool = (tool: string) => {
    setEnabledTools((prev) => {
      const next = prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool];
      syncYamlTools(next);
      return next;
    });
  };

  // Toggle all tools in a category on/off
  const toggleCategory = (tools: string[], checked: boolean) => {
    setEnabledTools((prev) => {
      const other = prev.filter((t) => !tools.includes(t));
      const next = checked ? [...other, ...tools] : other;
      syncYamlTools(next);
      return next;
    });
  };

  // Sync enabled_tools change into YAML editor content
  const syncYamlTools = (tools: string[]) => {
    setYamlContent((prev) => {
      if (!prev) return prev;
      return patchYamlWithFormValues(prev, { enabled_tools: tools });
    });
  };

  const handleFormValuesChange = (changed: Record<string, unknown>) => {
    const interesting = ['epochs', 'batch_size', 'workers', 'accumulation', 'seed',
      'learning_rate', 'min_learning_rate', 'lr_scheduler', 'update_mode',
      'slow_update', 'slow_update_samples', 'meta_skill', 'longitudinal_pair_policy',
      'minibatch_size', 'merge_batch_size', 'max_analyst_rounds', 'failure_only',
      'reasoning_effort', 'rewrite_reasoning_effort', 'rewrite_max_completion_tokens',
      'sel_env_num', 'test_env_num', 'eval_test', 'exec_timeout',
      'memory_algorithm'];
    if (!interesting.some((k) => k in changed)) return;
    setYamlContent((prev) => {
      if (!prev) return prev;
      const all = { ...form.getFieldsValue(), enabled_tools: enabledTools };
      return patchYamlWithFormValues(prev, all);
    });
  };

  const tabItems = [
    {
      key: 'data',
      label: 'Data',
      children: (
        <div>
          {copyFrom && hasSourceData ? (
            <Alert
              type="success"
              showIcon
              style={{ marginBottom: 16 }}
              message={`Data already uploaded from source task (${copyFrom}). It will be copied automatically when creating this task.`}
              description="You can still upload new files below to override, but typically no action is needed."
            />
          ) : null}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div>
              <p style={{ marginBottom: 2, fontSize: 11, color: '#aaa' }}>main data file</p>
              <Upload
                accept=".json,.jsonl,.xlsx"
                showUploadList={false}
                beforeUpload={(file) => { setMainFile(file as unknown as File); return false; }}
              >
                <Button icon={<CloudUploadOutlined />} size="small">
                  {mainFile ? mainFile.name : 'Upload'}
                </Button>
              </Upload>
              {mainFile && (
                <Button
                  type="link" size="small" danger icon={<DeleteOutlined />}
                  onClick={() => setMainFile(null)}
                  style={{ marginLeft: 8 }}
                >Remove</Button>
              )}
            </div>
            <div>
              <p style={{ marginBottom: 2, fontSize: 11, color: '#aaa' }}>test file</p>
              <Upload
                accept=".json,.jsonl,.xlsx"
                showUploadList={false}
                beforeUpload={(file) => { setTestFile(file as unknown as File); return false; }}
              >
                <Button icon={<CloudUploadOutlined />} size="small">
                  {testFile ? testFile.name : 'Upload'}
                </Button>
              </Upload>
              {testFile && (
                <Button
                  type="link" size="small" danger icon={<DeleteOutlined />}
                  onClick={() => setTestFile(null)}
                  style={{ marginLeft: 8 }}
                >Remove</Button>
              )}
            </div>
          </div>

          {mainFile && (
            <Alert
              type={hasTestFile ? 'warning' : 'info'}
              showIcon
              style={{ marginBottom: 16 }}
              message={
                hasTestFile
                  ? `Test file provided — main file will be split into train:val only (${form.getFieldValue('train_ratio') ?? 7}:${form.getFieldValue('val_ratio') ?? 2}). Test ratio ignored.`
                  : `Main file will be auto-split into train:val:test = ${form.getFieldValue('train_ratio') ?? 7}:${form.getFieldValue('val_ratio') ?? 2}:${form.getFieldValue('test_ratio') ?? 1}`
              }
            />
          )}

          <Divider orientation="left" plain>split ratio</Divider>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <Form.Item name="train_ratio" label="train ratio">
              <InputNumber min={0} step={1} precision={0} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item name="val_ratio" label="val ratio">
              <InputNumber min={0} step={1} precision={0} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item name="test_ratio" label="test ratio" tooltip={hasTestFile ? 'Ignored when test file is provided' : ''}>
              <InputNumber min={0} step={1} precision={0} style={{ width: '100%' }} disabled={hasTestFile} />
            </Form.Item>
            <Form.Item name="split_seed" label="split seed">
              <InputNumber min={0} precision={0} style={{ width: '100%' }} />
            </Form.Item>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: scorerMode === 'custom' ? '1fr 1fr' : '1fr', gap: 16, alignItems: 'end' }}>
            <Form.Item name="scorer_mode" label="default scorer" style={{ marginBottom: 16 }}>
              <Select
                options={[
                  { value: 'exact_match', label: 'exact_match' },
                  { value: 'llm_judge', label: 'llm_judge' },
                  { value: 'custom', label: 'custom (requires custom-scorer.py)' },
                ]}
              />
            </Form.Item>

            {scorerMode === 'custom' && (
              <div style={{ marginBottom: 16 }}>
                <p style={{ marginBottom: 2, fontSize: 11, color: '#aaa' }}>custom script</p>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <Upload
                    accept=".py"
                    showUploadList={false}
                    beforeUpload={(file) => { setScorerFile(file as unknown as File); return false; }}
                  >
                    <Button icon={<CloudUploadOutlined />} size="small">
                      {scorerFile ? scorerFile.name : 'Upload'}
                    </Button>
                  </Upload>
                  {scorerFile && (
                    <Button
                      type="link" size="small" danger icon={<DeleteOutlined />}
                      onClick={() => setScorerFile(null)}
                      style={{ marginLeft: 8 }}
                    >Remove</Button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      ),
    },
    {
      key: 'yaml',
      label: 'YAML',
      children: (
        <div>
          <Alert
            type="info" showIcon
            message="Edit skillopt.yaml directly below. Form parameters above will be applied on top when the task is created."
            style={{ marginBottom: 12 }}
          />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <p style={{ marginBottom: 8, fontWeight: 600 }}>Raw YAML Editor</p>
              <Input.TextArea
                value={yamlContent}
                onChange={(e) => setYamlContent(e.target.value)}
                rows={22}
                style={{ fontFamily: 'monospace', fontSize: 13, lineHeight: 1.6 }}
                placeholder="YAML configuration will be loaded here..."
              />
            </div>
            <div>
              <p style={{ marginBottom: 8, fontWeight: 600 }}>Parameter Reference</p>
              <YamlConfigViewer content={yamlContent} showRawToggle={false} maxHeight={520} />
            </div>
          </div>
        </div>
      ),
    },
  ];

  return (
    <Card title={copySource ? `Create New Task (copy from: ${copySource})` : 'Create New Task'}>
      {copyLoading && <Spin style={{ display: 'block', marginBottom: 24 }} tip="Loading source config..." />}
      {copySource && (
        <Alert
          type="info" showIcon
          style={{ marginBottom: 16 }}
          message={`Pre-filled configuration from task "${copySource}"`}
          description="A new task name is required. All other settings are pre-filled from the source task. YAML config will be copied to the new task."
          closable
        />
      )}
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        onValuesChange={handleFormValuesChange}
        initialValues={{
          algorithm: algorithms[0] || 'skillopt',
          epochs: 3,
          batch_size: 5,
          workers: 0,
          accumulation: 1,
          seed: 42,
          learning_rate: 4,
          min_learning_rate: 2,
          lr_scheduler: 'cosine',
          update_mode: 'patch',
          slow_update: true,
          slow_update_samples: 20,
          meta_skill: true,
          longitudinal_pair_policy: 'mixed',
          minibatch_size: 8,
          merge_batch_size: 8,
          max_analyst_rounds: 3,
          failure_only: false,
          reasoning_effort: 'medium',
          rewrite_reasoning_effort: '',
          rewrite_max_completion_tokens: 64000,
          sel_env_num: 0,
          test_env_num: 0,
          eval_test: true,
          exec_timeout: 120,
          memory_algorithm: 'null',
          train_ratio: 7,
          val_ratio: 2,
          test_ratio: 1,
          split_seed: 42,
          scorer_mode: 'exact_match',
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Form.Item name="name" label="task name" rules={[{ required: true, message: 'Required' }]}>
            <Input placeholder="my-training-task" />
          </Form.Item>
          <Form.Item name="description" label="description">
            <Input placeholder="Optional description..." />
          </Form.Item>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Form.Item name="algorithm" label="algorithm" rules={[{ required: true }]}>
            <Select options={algorithms.map((a) => ({ value: a, label: a }))} />
          </Form.Item>
          <Form.Item
            name="skill_path"
            label="initial skill path (optional)"
            tooltip={copyFrom && hasSourceSkill ? 'Skill will be copied from source task automatically' : undefined}
          >
            <Input
              placeholder="/path/to/skill.md"
              disabled={Boolean(copyFrom && hasSourceSkill)}
            />
          </Form.Item>
        </div>

        <Divider orientation="left" plain style={{ margin: '12px 0 8px' }}>train</Divider>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <Form.Item name="epochs" label="epochs">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="batch_size" label="batch size">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="workers" label="workers (0 = auto 80% maxConcurrency)">
            <InputNumber min={0} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="accumulation" label="accumulation" tooltip="每 N 个 batch 后执行一次参数更新，1 = 不累积">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="seed" label="seed">
            <InputNumber min={0} precision={0} style={{ width: '100%' }} />
          </Form.Item>
        </div>

        <Divider orientation="left" plain style={{ margin: '12px 0 8px' }}>optimizer</Divider>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <Form.Item name="learning_rate" label="learning rate (max edits)">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="min_learning_rate" label="min learning rate" tooltip="衰减调度器的下限">
            <InputNumber min={0} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="lr_scheduler" label="lr scheduler">
            <Select
              options={[
                { value: 'constant', label: 'Constant' },
                { value: 'linear', label: 'Linear' },
                { value: 'cosine', label: 'Cosine' },
                { value: 'autonomous', label: 'Autonomous' },
              ]}
            />
          </Form.Item>
          <Form.Item name="update_mode" label="update mode">
            <Select
              options={[
                { value: 'patch', label: 'Patch' },
                { value: 'rewrite_from_suggestions', label: 'Rewrite from Suggestions' },
                { value: 'full_rewrite_minibatch', label: 'Full Rewrite (Minibatch)' },
              ]}
            />
          </Form.Item>
          <Form.Item name="longitudinal_pair_policy" label="longitudinal pair policy" tooltip="纵向对比策略">
            <Select
              options={[
                { value: 'mixed', label: 'Mixed' },
                { value: 'changed', label: 'Changed' },
                { value: 'unchanged', label: 'Unchanged' },
              ]}
            />
          </Form.Item>
        </div>

        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
          <Form.Item name="slow_update" label="slow update" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name="slow_update_samples" label="slow update samples" tooltip="用于比较的历史样本数">
            <InputNumber min={1} precision={0} style={{ width: 120 }} />
          </Form.Item>
          <Form.Item name="meta_skill" label="meta skill" valuePropName="checked">
            <Switch />
          </Form.Item>
        </div>

        <Divider orientation="left" plain style={{ margin: '12px 0 8px' }}>gradient</Divider>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Form.Item name="minibatch_size" label="minibatch size" tooltip="每次分析处理的轨迹数">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="merge_batch_size" label="merge batch size" tooltip="层次合并的批大小">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="max_analyst_rounds" label="max analyst rounds" tooltip="最大分析迭代次数">
            <InputNumber min={1} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="failure_only" label="failure only" valuePropName="checked" tooltip="只分析失败的轨迹">
            <Switch />
          </Form.Item>
        </div>

        <Divider orientation="left" plain style={{ margin: '12px 0 8px' }}>model</Divider>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Form.Item name="reasoning_effort" label="reasoning effort">
            <Select
              options={[
                { value: 'low', label: 'Low' },
                { value: 'medium', label: 'Medium' },
                { value: 'high', label: 'High' },
              ]}
            />
          </Form.Item>
          <Form.Item name="rewrite_reasoning_effort" label="rewrite reasoning effort" tooltip="单独控制 rewrite 阶段的努力程度，留空表示使用 reasoning_effort">
            <Select
              allowClear
              placeholder="(same as reasoning_effort)"
              options={[
                { value: 'low', label: 'Low' },
                { value: 'medium', label: 'Medium' },
                { value: 'high', label: 'High' },
              ]}
            />
          </Form.Item>
          <Form.Item name="rewrite_max_completion_tokens" label="rewrite max tokens" tooltip="rewrite 阶段的输出 token 限制">
            <InputNumber min={1000} step={1000} precision={0} style={{ width: '100%' }} />
          </Form.Item>
        </div>

        <Divider orientation="left" plain style={{ margin: '12px 0 8px' }}>memory &amp; tools</Divider>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Form.Item name="memory_algorithm" label="memory algorithm" tooltip="Select memory backend or NULL to disable">
            <Select
              options={[
                { value: 'null', label: 'NULL (disabled)' },
                ...memoryAlgorithms.map((a) => ({ value: a, label: a })),
              ]}
            />
          </Form.Item>
          <Form.Item label="enabled tools" tooltip="选中需要启用的工具，未选中的分类默认不参与训练">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {toolCategories.map((cat) => {
                const catChecked = cat.tools.filter((t) => enabledTools.includes(t));
                const allChecked = catChecked.length === cat.tools.length;
                const someChecked = catChecked.length > 0 && !allChecked;
                return (
                  <div key={cat.key} style={{ border: '1px solid #d9d9d9', borderRadius: 6, padding: '8px 12px' }}>
                    {/* Category header */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <input
                        type="checkbox"
                        ref={(el) => { if (el) el.indeterminate = someChecked; }}
                        checked={allChecked}
                        onChange={(e) => toggleCategory(cat.tools, e.target.checked)}
                        style={{ width: 16, height: 16, cursor: 'pointer', accentColor: '#1677ff' }}
                      />
                      <span style={{ fontWeight: 600, fontSize: 13 }}>{cat.label}</span>
                      {cat.default_excluded && (
                        <span style={{ fontSize: 11, color: '#999', marginLeft: 4 }}>(默认关闭)</span>
                      )}
                    </div>
                    {/* Tool checkboxes */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px', paddingLeft: 24 }}>
                      {cat.tools.map((t) => (
                        <label
                          key={t}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: 13, userSelect: 'none' }}
                        >
                          <input
                            type="checkbox"
                            checked={enabledTools.includes(t)}
                            onChange={() => toggleTool(t)}
                            style={{ width: 16, height: 16, cursor: 'pointer', accentColor: '#1677ff' }}
                          />
                          {t}
                        </label>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </Form.Item>
        </div>

        <Divider orientation="left" plain style={{ margin: '12px 0 8px' }}>evaluation & env</Divider>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Form.Item name="eval_test" label="eval test" valuePropName="checked" tooltip="是否在测试集上评估">
            <Switch />
          </Form.Item>
          <Form.Item name="sel_env_num" label="sel env num" tooltip="用于门控验证的环境数量，0=全部">
            <InputNumber min={0} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="test_env_num" label="test env num" tooltip="用于最终测试的环境数量，0=全部">
            <InputNumber min={0} precision={0} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="exec_timeout" label="exec timeout (s)" tooltip="单次目标模型调用的超时时间（秒）">
            <InputNumber min={10} precision={0} style={{ width: '100%' }} />
          </Form.Item>
        </div>

        <Card type="inner" title="Additional Settings" style={{ marginBottom: 16 }}>
          <Tabs items={tabItems} />
        </Card>

        <Space>
          <Button type="primary" htmlType="submit" loading={submitting} disabled={toolsLoading} size="large">
            {toolsLoading ? 'Loading tools...' : 'Create Task'}
          </Button>
          <Button onClick={handleReset} size="large">
            Reset
          </Button>
        </Space>
      </Form>
    </Card>
  );
};
