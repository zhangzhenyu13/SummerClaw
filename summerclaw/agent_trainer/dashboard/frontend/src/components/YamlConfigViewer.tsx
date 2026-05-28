/** YamlConfigViewer — renders parsed YAML with parameter descriptions.
 *
 *  Parses the skillopt.yaml structured format (sections → key: value)
 *  and displays each parameter with its description as a Tooltip.
 *  Supports both readonly and raw-source toggle modes.
 */

import React, { useMemo, useState } from 'react';
import {
  Collapse, Descriptions, Tag, Tooltip, Typography, Switch, Space, Empty,
} from 'antd';
import {
  InfoCircleOutlined, CodeOutlined, EyeOutlined,
  LockOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

// ---------------------------------------------------------------------------
// Parameter description schema
// ---------------------------------------------------------------------------

interface ParamDesc {
  label: string;
  desc: string;
  recommend?: string;
  locked?: boolean;     // system-managed, do not edit
}

interface SectionDesc {
  label: string;
  desc: string;
  params: Record<string, ParamDesc>;
}

const YAML_SCHEMA: Record<string, SectionDesc> = {
  train: {
    label: '训练参数 (Train)',
    desc: '控制训练循环的核心参数',
    params: {
      num_epochs: {
        label: '训练轮数',
        desc: '完整遍历训练集的次数',
        recommend: '3-10，数据量大时可适当增加',
      },
      batch_size: {
        label: '批处理大小',
        desc: '每次训练迭代处理的样本数',
        recommend: '20-50，取决于显存和数据量',
      },
      workers: {
        label: '并发工作数',
        desc: '控制所有阶段（Rollout/Reflect/Aggregate/Evaluate）的 LLM 并发调用数。设为 0 时自动推导为系统 maxConcurrency 的 80%',
        recommend: '0（自动）或 8-32，可在 gradient 分区单独覆盖各阶段',
      },
      accumulation: {
        label: '梯度累积步数',
        desc: '每 N 个 batch 后执行一次参数更新，用于模拟大 batch 训练',
        recommend: '设为 1 表示不累积',
      },
      seed: {
        label: '随机种子',
        desc: '保证训练可复现的随机种子',
      },
    },
  },
  optimizer: {
    label: '优化器参数 (Optimizer)',
    desc: '控制 SkillOpt 技能优化器的行为',
    params: {
      learning_rate: {
        label: '学习率 (最大编辑数)',
        desc: '每步最多应用的编辑操作数',
        recommend: '2-6，过高可能导致技能退化',
      },
      min_learning_rate: {
        label: '最小学习率',
        desc: '衰减调度器的下限值',
      },
      lr_scheduler: {
        label: '学习率调度策略',
        desc: 'constant: 固定学习率 | linear: 线性衰减 | cosine: 余弦退火 | autonomous: LLM 自主决定',
        recommend: 'cosine 通常效果最好',
      },
      skill_update_mode: {
        label: '技能更新模式',
        desc: 'patch: 增量编辑 | rewrite_from_suggestions: 基于建议重写 | full_rewrite_minibatch: 全量重写(实验性)',
        recommend: 'patch 稳定可靠',
      },
      use_slow_update: {
        label: '慢更新',
        desc: '跨 epoch 的长期技能指导开关',
      },
      slow_update_samples: {
        label: '慢更新样本数',
        desc: '用于比较的历史样本数量',
      },
      use_meta_skill: {
        label: '元技能',
        desc: '跨 epoch 的优化器记忆开关',
      },
      longitudinal_pair_policy: {
        label: '纵向对比策略',
        desc: 'mixed: 所有样本 | changed: 仅变化的样本 | unchanged: 仅稳定的样本',
        recommend: 'mixed（推荐）',
      },
    },
  },
  gradient: {
    label: '梯度分析参数 (Gradient)',
    desc: '控制 Reflect 阶段的轨迹分析',
    params: {
      minibatch_size: {
        label: '最小批大小',
        desc: '每次分析处理的轨迹数',
        recommend: '4-12，影响分析质量和速度',
      },
      merge_batch_size: {
        label: '合并批大小',
        desc: '层次合并的批大小',
      },
      max_analyst_rounds: {
        label: '分析轮数',
        desc: '最大分析迭代次数',
      },
      analyst_workers: {
        label: 'Reflect 阶段并发数',
        desc: '覆盖 train.workers 作为 Reflect 阶段的 LLM 并行度，未设置时继承 train.workers',
        recommend: '留空（继承 workers）或单独设置',
      },
      aggregate_workers: {
        label: 'Aggregate 阶段并发数',
        desc: '覆盖 train.workers 作为 Aggregate 阶段的 LLM 并行度，未设置时继承 train.workers',
        recommend: '留空（继承 workers）或单独设置',
      },
      evaluate_workers: {
        label: 'Evaluate 阶段并发数',
        desc: '覆盖 train.workers 作为 Evaluate 阶段的 rollout 并行度，未设置时继承 train.workers',
        recommend: '留空（继承 workers）或单独设置',
      },
      failure_only: {
        label: '仅分析失败',
        desc: '只分析失败的轨迹（加速但可能遗漏信息）',
      },
    },
  },
  model: {
    label: '推理参数 (Model)',
    desc: '控制 LLM 调用的推理行为',
    params: {
      reasoning_effort: {
        label: '推理努力程度',
        desc: 'low: 快速响应 | medium: 平衡 | high: 深度思考',
        recommend: 'medium（推荐）',
      },
      rewrite_reasoning_effort: {
        label: '重写推理努力',
        desc: '单独控制 rewrite 阶段的努力程度，留空表示使用 reasoning_effort 的值',
      },
      rewrite_max_completion_tokens: {
        label: '重写最大 token 数',
        desc: 'rewrite 阶段的输出 token 限制',
      },
    },
  },
  evaluation: {
    label: '评估参数 (Evaluation)',
    desc: '控制技能验证和门控',
    params: {
      use_gate: {
        label: '门控验证',
        desc: '验证候选技能是否优于当前技能（必须启用）',
        locked: true,
      },
      sel_env_num: {
        label: '选择环境数',
        desc: '用于门控验证的环境数量，0 表示全部',
      },
      test_env_num: {
        label: '测试环境数',
        desc: '用于最终测试的环境数量，0 表示全部',
      },
      eval_test: {
        label: '评估测试集',
        desc: '是否在测试集上评估',
      },
    },
  },
  env: {
    label: '环境参数 (Env)',
    desc: '环境与数据路径配置，大部分由 Dashboard 自动设置',
    params: {
      name: {
        label: '环境名称',
        desc: '由系统自动设置',
        locked: true,
      },
      skill_init: {
        label: '初始技能路径',
        desc: '由 Dashboard 设置或自动检测',
        locked: true,
      },
      split_mode: {
        label: '数据切分模式',
        desc: 'ratio: 按比例切分 | split_dir: 使用预切分目录',
      },
      split_ratio: {
        label: '切分比例',
        desc: 'train:val:test 比例（仅 split_mode=ratio 时生效）',
      },
      split_seed: {
        label: '切分随机种子',
        desc: '数据切分的随机种子',
      },
      split_dir: {
        label: '切分目录',
        desc: '由 Dashboard 数据上传自动设置',
        locked: true,
      },
      data_path: {
        label: '数据路径',
        desc: '由 Dashboard 数据上传自动设置',
        locked: true,
      },
      exec_timeout: {
        label: '执行超时',
        desc: '单次目标模型调用的超时时间（秒）',
      },
      memory_algorithm: {
        label: '记忆算法',
        desc: '训练时使用的记忆算法，null 表示禁用记忆',
        recommend: 'naive_memory 或 null（禁用）',
      },
      enabled_tools: {
        label: '启用工具',
        desc: 'Rollout 阶段可用的工具列表，空列表表示使用全部默认工具',
      },
    },
  },
};

// ---------------------------------------------------------------------------
// Lightweight YAML parser for the structured skillopt format
// ---------------------------------------------------------------------------

export interface ParsedParam {
  key: string;
  value: string;
  comment: string;
}

export interface ParsedSection {
  name: string;
  params: ParsedParam[];
}

export function parseStructuredYaml(raw: string): ParsedSection[] {
  const lines = raw.split('\n');
  const sections: ParsedSection[] = [];
  let current: ParsedSection | null = null;
  // Tracks a key whose value is a YAML block list ("- item" lines).
  // We accumulate items and commit when the next key/section/EOF arrives.
  let pendingListKey: string | null = null;
  let pendingListItems: string[] = [];

  const commitPendingList = () => {
    if (pendingListKey && current) {
      current.params.push({
        key: pendingListKey,
        value: '[' + pendingListItems.map((s) => `"${s}"`).join(', ') + ']',
        comment: '',
      });
    }
    pendingListKey = null;
    pendingListItems = [];
  };

  for (const line of lines) {
    const trimmed = line.trimEnd();

    // Skip empty / pure-comment / header lines
    if (!trimmed || trimmed.startsWith('#')) continue;

    // Section header (no leading whitespace, ends with ':')
    if (!line.startsWith(' ') && !line.startsWith('\t') && trimmed.endsWith(':')) {
      commitPendingList();
      const name = trimmed.slice(0, -1).trim();
      current = { name, params: [] };
      sections.push(current);
      continue;
    }

    // Key: value inside a section
    if (current && (line.startsWith('  ') || line.startsWith('\t'))) {
      // YAML block list item: "- <value>" (possibly indented further)
      const listItemMatch = trimmed.match(/^\s*-\s+(.*)$/);
      if (listItemMatch && pendingListKey) {
        let item = (listItemMatch[1] ?? '').trim();
        if ((item.startsWith('"') && item.endsWith('"')) ||
            (item.startsWith("'") && item.endsWith("'"))) {
          item = item.slice(1, -1);
        }
        pendingListItems.push(item);
        continue;
      }
      // Not a list item — flush any in-progress list before parsing a new key.
      commitPendingList();

      // Strip inline comment
      const commentMatch = trimmed.match(/^(\s*[\w.]+:\s*.*?)\s*#(.*)$/);
      let kvStr = trimmed.trim();
      let comment = '';
      if (commentMatch && commentMatch[1] && commentMatch[2]) {
        kvStr = commentMatch[1].trim();
        comment = commentMatch[2].trim();
      }

      const colonIdx = kvStr.indexOf(':');
      if (colonIdx > 0) {
        const key = kvStr.slice(0, colonIdx).trim();
        let value = kvStr.slice(colonIdx + 1).trim();
        // Strip surrounding quotes
        if ((value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'"))) {
          value = value.slice(1, -1);
        }
        if (value) {
          current.params.push({ key, value, comment });
        } else {
          // Value-less key — next "- item" lines will build a block list.
          pendingListKey = key;
          pendingListItems = [];
        }
      }
    }
  }
  commitPendingList();

  return sections;
}

// ---------------------------------------------------------------------------
// Value formatter
// ---------------------------------------------------------------------------

function formatValue(val: string): React.ReactNode {
  if (val === 'true') return <Tag color="green">true</Tag>;
  if (val === 'false') return <Tag color="red">false</Tag>;
  if (val === '' || val === '""' || val === "''") return <Text type="secondary">(empty)</Text>;
  if (val === '[]') return <Text type="secondary">[] (all defaults)</Text>;
  // JSON-style list produced by the parser: ["a", "b", "c"]
  if (val.startsWith('[') && val.endsWith(']')) {
    const inner = val.slice(1, -1).trim();
    if (!inner) return <Text type="secondary">[] (all defaults)</Text>;
    const items = inner
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => (s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))
        ? s.slice(1, -1)
        : s);
    return (
      <Space size={[4, 4]} wrap>
        {items.map((item, idx) => (
          <Tag key={idx} color="geekblue">{item}</Tag>
        ))}
      </Space>
    );
  }
  if (/^-?\d+$/.test(val)) return <Tag color="blue">{val}</Tag>;
  if (/^-?\d+\.\d+$/.test(val)) return <Tag color="cyan">{val}</Tag>;
  return <Text code>{val}</Text>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface YamlConfigViewerProps {
  /** Raw YAML content string */
  content: string;
  /** If true, show raw YAML toggle button */
  showRawToggle?: boolean;
  /** Max height for the container (px) */
  maxHeight?: number;
}

export const YamlConfigViewer: React.FC<YamlConfigViewerProps> = ({
  content,
  showRawToggle = true,
  maxHeight = 650,
}) => {
  const [showRaw, setShowRaw] = useState(false);

  const sections = useMemo(() => parseStructuredYaml(content || ''), [content]);

  if (!sections.length) {
    return <Empty description="No structured YAML content" />;
  }

  // Build collapse items
  const collapseItems = sections.map((sec) => {
    const schema = YAML_SCHEMA[sec.name];
    const sectionLabel = schema?.label || sec.name;
    const sectionDesc = schema?.desc || '';

    const header = (
      <Space>
        <strong>{sectionLabel}</strong>
        <Text type="secondary" style={{ fontSize: 12 }}>{sectionDesc}</Text>
        <Tag>{sec.params.length}</Tag>
      </Space>
    );

    const rows = sec.params.map((p) => {
      const desc = schema?.params?.[p.key];
      const tooltipContent = desc ? (
        <div style={{ maxWidth: 320 }}>
          <div><strong>{desc.label}</strong></div>
          <div style={{ marginTop: 4 }}>{desc.desc}</div>
          {desc.recommend && (
            <div style={{ marginTop: 4, color: '#95de64' }}>
              推荐: {desc.recommend}
            </div>
          )}
          {desc.locked && (
            <div style={{ marginTop: 4, color: '#ffa39e' }}>
              🔒 系统自动管理，请勿手动修改
            </div>
          )}
        </div>
      ) : (
        <div>{p.comment || 'No description available'}</div>
      );

      const paramLabel = desc?.label || p.key;
      const isLocked = desc?.locked;

      return (
        <Descriptions.Item
          key={p.key}
          label={
            <Tooltip title={tooltipContent} placement="left" mouseEnterDelay={0.2}>
              <Space size={4}>
                {isLocked && <LockOutlined style={{ color: '#faad14', fontSize: 12 }} />}
                <span>{paramLabel}</span>
                <InfoCircleOutlined style={{ color: '#bbb', fontSize: 11 }} />
                {p.key !== paramLabel && (
                  <Text type="secondary" style={{ fontSize: 11 }}>({p.key})</Text>
                )}
              </Space>
            </Tooltip>
          }
        >
          {formatValue(p.value)}
        </Descriptions.Item>
      );
    });

    return {
      key: sec.name,
      label: header,
      children: (
        <Descriptions
          bordered
          column={1}
          size="small"
          labelStyle={{ width: 220, fontWeight: 500 }}
          contentStyle={{ minWidth: 180 }}
        >
          {rows}
        </Descriptions>
      ),
    };
  });

  return (
    <div>
      {showRawToggle && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
          <Space>
            {showRaw ? <CodeOutlined /> : <EyeOutlined />}
            <span style={{ fontSize: 13 }}>{showRaw ? 'Raw YAML' : 'Structured View'}</span>
            <Switch
              size="small"
              checked={showRaw}
              onChange={setShowRaw}
            />
          </Space>
        </div>
      )}

      {showRaw ? (
        <pre
          style={{
            background: '#f5f5f5',
            border: '1px solid #d9d9d9',
            borderRadius: 6,
            padding: 16,
            margin: 0,
            fontFamily: 'monospace',
            fontSize: 13,
            lineHeight: 1.6,
            maxHeight,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {content}
        </pre>
      ) : (
        <div style={{ maxHeight, overflow: 'auto' }}>
          <Collapse
            defaultActiveKey={sections.map((s) => s.name)}
            items={collapseItems}
            bordered
            size="small"
          />
        </div>
      )}
    </div>
  );
};
