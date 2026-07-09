import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BookOpen,
  Bot,
  GraduationCap,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const API_BASE_URL = (import.meta.env.VITE_DIRECT_API_BASE_URL || "").replace(/\/$/, "");
const COLORS = [
  "#2563eb",
  "#16a34a",
  "#f97316",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#64748b",
  "#ca8a04",
  "#0f766e",
  "#dc2626",
  "#4f46e5",
  "#65a30d",
  "#c026d3",
  "#ea580c",
  "#0284c7",
  "#be123c",
  "#9333ea",
  "#059669",
  "#b45309",
  "#475569",
  "#1d4ed8",
  "#15803d",
  "#e11d48",
  "#7e22ce",
];

function colorAt(index) {
  return COLORS[index % COLORS.length];
}

function apiUrl(path) {
  if (!API_BASE_URL) return path;
  return `${API_BASE_URL}${path}`;
}

async function apiGet(path) {
  const response = await fetch(apiUrl(path));
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `请求失败：${response.status}`);
  }
  return response.json();
}

async function apiPost(path, body) {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `请求失败：${response.status}`);
  }
  return response.json();
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("zh-CN");
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(4);
}

const COLUMN_LABELS = {
  rank: "隐向量维度",
  maxIter: "最大迭代次数",
  max_iter: "最大迭代次数",
  regParam: "正则化参数",
  reg_param: "正则化参数",
  topN: "推荐数量",
  top_n: "推荐数量",
  rmse: "均方根误差",
  coverage: "覆盖率",
  diversity: "多样性",
  popular_recommendation_ratio: "热门推荐占比",
  trained_at: "训练时间",
  rank_position: "排名",
  course_id: "课程ID",
  course_name: "课程名称",
  category: "课程类别",
  teacher: "教师/来源",
  similar_user_count: "相似用户数",
  avg_preference: "平均偏好",
  recommendation_score: "推荐分数",
  user_id: "学习者ID",
  learn_count: "学习次数",
  size: "出现次数",
  count: "数量",
};

function labelForColumn(column) {
  return COLUMN_LABELS[column] || column;
}

function useApi(path, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    setError("");
    apiGet(path)
      .then((result) => {
        if (!ignore) setData(result);
      })
      .catch((err) => {
        if (!ignore) setError(err.message);
      })
      .finally(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, deps);

  return { data, loading, error, setData };
}

function LoadingBlock() {
  return (
    <div className="state-block">
      <Loader2 className="spin" size={20} />
      <span>正在读取本地后端数据</span>
    </div>
  );
}

function ErrorBlock({ message }) {
  return <div className="error-block">{message}</div>;
}

function Metric({ label, value, tone = "blue" }) {
  return (
    <div className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Section({ title, icon: Icon, children, actions }) {
  return (
    <section className="section">
      {(title || actions) && (
        <div className="section-head">
          {title && (
            <div>
              <h2>
                {Icon && <Icon size={20} />}
                {title}
              </h2>
            </div>
          )}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

function MarkdownMessage({ content }) {
  const blocks = parseMarkdownBlocks(content || "");
  return (
    <div className="markdown-body">
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          const Tag = `h${block.level}`;
          return <Tag key={index}>{renderInlineMarkdown(block.text)}</Tag>;
        }
        if (block.type === "list") {
          const Tag = block.ordered ? "ol" : "ul";
          return (
            <Tag key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>{renderInlineMarkdown(item)}</li>
              ))}
            </Tag>
          );
        }
        if (block.type === "code") {
          return <pre key={index}><code>{block.text}</code></pre>;
        }
        if (block.type === "quote") {
          return <blockquote key={index}>{renderInlineMarkdown(block.text)}</blockquote>;
        }
        return <p key={index}>{renderInlineMarkdown(block.text)}</p>;
      })}
    </div>
  );
}

function parseMarkdownBlocks(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (line.trim().startsWith("```")) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      blocks.push({ type: "code", text: codeLines.join("\n") });
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      index += 1;
      continue;
    }

    const list = line.match(/^(\s*)([-*]|\d+\.)\s+(.+)$/);
    if (list) {
      const ordered = /^\d+\./.test(list[2]);
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^(\s*)([-*]|\d+\.)\s+(.+)$/);
        if (!item || /^\d+\./.test(item[2]) !== ordered) break;
        items.push(item[3]);
        index += 1;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }

    if (line.trim().startsWith(">")) {
      const quotes = [];
      while (index < lines.length && lines[index].trim().startsWith(">")) {
        quotes.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      blocks.push({ type: "quote", text: quotes.join(" ") });
      continue;
    }

    const paragraphs = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].trim().startsWith("```") &&
      !/^(#{1,4})\s+/.test(lines[index]) &&
      !/^(\s*)([-*]|\d+\.)\s+/.test(lines[index]) &&
      !lines[index].trim().startsWith(">")
    ) {
      paragraphs.push(lines[index].trim());
      index += 1;
    }
    blocks.push({ type: "paragraph", text: paragraphs.join(" ") });
  }

  return blocks.length ? blocks : [{ type: "paragraph", text: "" }];
}

function renderInlineMarkdown(text) {
  const parts = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
    const token = match[0];
    if (token.startsWith("`")) {
      parts.push(<code key={parts.length}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**")) {
      parts.push(<strong key={parts.length}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<em key={parts.length}>{token.slice(1, -1)}</em>);
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

function PieLabel({ cx, cy, midAngle, outerRadius, percent, value, name, index }) {
  if (!percent || percent < 0.015) return null;

  const radius = outerRadius + 22;
  const angle = (-midAngle * Math.PI) / 180;
  const x = cx + radius * Math.cos(angle);
  const y = cy + radius * Math.sin(angle);
  const anchor = x > cx ? "start" : "end";

  return (
    <text x={x} y={y} fill={colorAt(index)} textAnchor={anchor} dominantBaseline="central" className="pie-label">
      {`${name || "未分类"} ${formatNumber(value)}`}
    </text>
  );
}

function CategoryLegend({ rows, valueKey, className = "" }) {
  if (!rows?.length) return null;

  return (
    <div className={`category-legend ${className}`.trim()} aria-label="类别索引">
      {rows.map((row, index) => (
        <div className="legend-item" key={`${row.category || "未分类"}-${index}`}>
          <span className="legend-swatch" style={{ backgroundColor: colorAt(index) }} />
          <span className="legend-index">{index + 1}</span>
          <span className="legend-name">{row.category || "未分类"}</span>
          <span className="legend-value">{formatNumber(row[valueKey])}</span>
        </div>
      ))}
    </div>
  );
}

function normalizeHistogramData(histogram) {
  if (Array.isArray(histogram)) {
    return {
      rows: histogram,
      tail: null,
    };
  }
  return {
    rows: histogram?.rows || [],
    tail: histogram?.tail || null,
  };
}

function buildOverviewInsights(summary, popularCourses, categoryDistribution) {
  const topCourse = popularCourses?.[0];
  const top3Count = (popularCourses || []).slice(0, 3).reduce((sum, course) => sum + Number(course.learn_count || 0), 0);
  const top10Count = (popularCourses || []).reduce((sum, course) => sum + Number(course.learn_count || 0), 0);
  const top3Share = top10Count ? `${((top3Count / top10Count) * 100).toFixed(1)}%` : "-";
  const topCategory = categoryDistribution?.[0];
  const categoryTotal = (categoryDistribution || []).reduce((sum, item) => sum + Number(item.interaction_count || 0), 0);
  const topCategoryShare = categoryTotal && topCategory ? `${((Number(topCategory.interaction_count || 0) / categoryTotal) * 100).toFixed(1)}%` : "-";
  const interactionsPerUser = summary.user_count ? (Number(summary.interaction_count || 0) / Number(summary.user_count)).toFixed(1) : "-";
  const interactionsPerCourse = summary.course_count ? (Number(summary.interaction_count || 0) / Number(summary.course_count)).toFixed(1) : "-";

  const notes = [];
  if (topCourse) {
    notes.push(`热门课程首位是「${topCourse.course_name}」，学习次数达到 ${formatNumber(topCourse.learn_count)}，说明基础课或通用能力类内容对学习者吸引力更强。`);
  }
  if (topCategory) {
    notes.push(`${topCategory.category || "未分类"} 是当前交互最集中的类别，占类别交互总量约 ${topCategoryShare}，推荐结果需要兼顾该类别优势与其他类别覆盖。`);
  }
  notes.push(`Top 3 热门课程占 Top 10 学习次数的 ${top3Share}，可用来判断热门内容是否过于集中。`);
  notes.push(`平均每位用户约产生 ${interactionsPerUser} 条交互，平均每门课程约获得 ${interactionsPerCourse} 条交互，整体数据可支撑协同过滤发现稳定偏好。`);

  return {
    stats: [
      { label: "Top 3 / Top 10", value: top3Share },
      { label: "首位类别占比", value: topCategoryShare },
      { label: "人均交互", value: interactionsPerUser },
      { label: "课均交互", value: interactionsPerCourse },
    ],
    notes,
  };
}

function OverviewInsight({ summary, popularCourses, categoryDistribution }) {
  const insight = buildOverviewInsights(summary, popularCourses, categoryDistribution);
  return (
    <div className="overview-insight">
      <div className="insight-head">
        <Sparkles size={17} />
        <strong>数据简析</strong>
      </div>
      <div className="insight-stats">
        {insight.stats.map((item) => (
          <div className="insight-stat" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
      <ul>
        {insight.notes.map((note, index) => (
          <li key={index}>{note}</li>
        ))}
      </ul>
    </div>
  );
}

function Overview() {
  const { data, loading, error } = useApi("/api/overview", []);
  if (loading) return <LoadingBlock />;
  if (error) return <ErrorBlock message={error} />;

  const summary = data.summary || {};
  const popularCourses = data.popular_courses || [];
  const categoryDistribution = data.category_distribution || [];
  const userCourseHistogram = normalizeHistogramData(data.user_course_histogram);
  const courseLearnHistogram = normalizeHistogramData(data.course_learn_histogram);
  return (
    <>
      <div className="metrics-grid">
        <Metric label="用户数量" value={formatNumber(summary.user_count)} tone="blue" />
        <Metric label="课程数量" value={formatNumber(summary.course_count)} tone="green" />
        <Metric label="交互记录" value={formatNumber(summary.interaction_count)} tone="orange" />
        <Metric label="平均偏好" value={summary.avg_preference ?? "-"} tone="purple" />
      </div>
      <div className="chart-grid">
        <Section title="热门课程 Top 10" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={[...popularCourses].reverse()} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" />
              <YAxis dataKey="course_name" type="category" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="learn_count" radius={[0, 4, 4, 0]}>
                {[...popularCourses].reverse().map((_, index) => (
                  <Cell key={index} fill={colorAt(index)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <OverviewInsight summary={summary} popularCourses={popularCourses} categoryDistribution={categoryDistribution} />
        </Section>
        <Section title="课程类别交互分布" icon={Activity}>
          <div className="pie-with-legend">
            <ResponsiveContainer width="100%" height={340}>
              <PieChart margin={{ top: 16, right: 88, bottom: 16, left: 88 }}>
                <Pie
                  data={categoryDistribution}
                  dataKey="interaction_count"
                  nameKey="category"
                  outerRadius={105}
                  innerRadius={54}
                  label={(props) => <PieLabel {...props} />}
                  labelLine
                >
                  {categoryDistribution.map((_, index) => (
                    <Cell key={index} fill={colorAt(index)} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <CategoryLegend rows={categoryDistribution} valueKey="interaction_count" className="overview-category-legend" />
          </div>
        </Section>
      </div>
      <div className="chart-grid">
        <Section title="用户选课数量分布" icon={Activity}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={userCourseHistogram.rows} margin={{ top: 8, right: 18, bottom: 34, left: 18 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="range"
                interval={1}
                tick={{ fontSize: 10 }}
                label={{ value: "用户选课数量区间（门）", position: "insideBottom", offset: -22 }}
              />
              <YAxis label={{ value: "用户数", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
        <Section title="课程被学习次数分布" icon={Activity}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={courseLearnHistogram.rows} margin={{ top: 12, right: 24, bottom: 34, left: 18 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="range"
                interval={1}
                tick={{ fontSize: 10 }}
                label={{ value: "课程被学习次数区间（次）", position: "insideBottom", offset: -22 }}
              />
              <YAxis label={{ value: "课程数", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#16a34a" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </Section>
      </div>
    </>
  );
}

function Training() {
  const { data, loading, error, setData } = useApi("/api/training/latest", []);
  const [form, setForm] = useState({ rank: 10, max_iter: 8, reg_param: 0.08, top_n: 8 });
  const [training, setTraining] = useState(false);
  const [message, setMessage] = useState("");

  async function handleTrain() {
    setTraining(true);
    setMessage("");
    try {
      const result = await apiPost("/api/training/train", form);
      const latest = await apiGet("/api/training/latest");
      setData(latest);
      setMessage(`训练完成：${result.metrics?.length || 0} 条指标已保存`);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setTraining(false);
    }
  }

  if (loading) return <LoadingBlock />;
  if (error) return <ErrorBlock message={error} />;
  const latest = data.latest || {};

  return (
    <>
      <Section
        title="模型训练"
        icon={RefreshCw}
        actions={
          <button className="primary" onClick={handleTrain} disabled={training}>
            {training ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            训练 ALS 模型
          </button>
        }
      >
        <div className="form-grid">
          <label>
            隐向量维度
            <input value={form.rank} type="number" min="2" max="50" onChange={(e) => setForm({ ...form, rank: Number(e.target.value) })} />
          </label>
          <label>
            最大迭代次数
            <input value={form.max_iter} type="number" min="3" max="30" onChange={(e) => setForm({ ...form, max_iter: Number(e.target.value) })} />
          </label>
          <label>
            正则化参数
            <input value={form.reg_param} type="number" step="0.01" min="0.001" max="1" onChange={(e) => setForm({ ...form, reg_param: Number(e.target.value) })} />
          </label>
          <label>
            推荐数量
            <input value={form.top_n} type="number" min="3" max="30" onChange={(e) => setForm({ ...form, top_n: Number(e.target.value) })} />
          </label>
        </div>
        {message && <div className="inline-note">{message}</div>}
      </Section>
      <div className="metrics-grid">
        <Metric label="均方根误差（RMSE）" value={latest.rmse ?? "-"} tone="blue" />
        <Metric label="覆盖率" value={formatPercent(latest.coverage)} tone="green" />
        <Metric label="多样性" value={latest.diversity ?? "-"} tone="orange" />
        <Metric label="热门推荐占比" value={formatPercent(latest.popular_recommendation_ratio)} tone="purple" />
      </div>
      <DataTable rows={data.history} />
    </>
  );
}

function Recommendations({ onAnalyzeCourse }) {
  const usersState = useApi("/api/recommendations/users", []);
  const coursesState = useApi("/api/courses?limit=5000", []);
  const [mode, setMode] = useState("user");
  const [userId, setUserId] = useState("");
  const [selectedCourses, setSelectedCourses] = useState([]);
  const [topN, setTopN] = useState(8);
  const [items, setItems] = useState([]);
  const [pinnedCourse, setPinnedCourse] = useState(null);
  const [error, setError] = useState("");
  const users = usersState.data?.users || [];
  const courses = coursesState.data?.courses || [];
  const courseById = useMemo(() => new Map(courses.map((course) => [course.course_id, course])), [courses]);
  const selectedCourseRows = useMemo(() => selectedCourses.map((id) => courseById.get(id)).filter(Boolean), [selectedCourses, courseById]);
  const chartItems = useMemo(() => [...items].reverse(), [items]);

  useEffect(() => {
    if (!userId && users.length) setUserId(users[0]);
  }, [users, userId]);

  async function loadRecommendations() {
    setError("");
    try {
      const path =
        mode === "user"
          ? `/api/recommendations/by-user?user_id=${encodeURIComponent(userId)}&top_n=${topN}`
          : `/api/recommendations/by-courses?${selectedCourses.map((id) => `course_ids=${encodeURIComponent(id)}`).join("&")}&top_n=${topN}`;
      const data = await apiGet(path);
      setItems(data.items || []);
      setPinnedCourse(null);
    } catch (err) {
      setError(err.message);
      setItems([]);
      setPinnedCourse(null);
    }
  }

  useEffect(() => {
    if (mode === "courses" && !selectedCourses.length) {
      setItems([]);
      setError("");
      return;
    }
    if ((mode === "user" && userId) || mode === "courses") {
      loadRecommendations();
    }
  }, [mode, userId, selectedCourses.join(","), topN]);

  if (usersState.loading || coursesState.loading) return <LoadingBlock />;
  if (usersState.error) return <ErrorBlock message={usersState.error} />;
  const recommendationCategories = categoryRows(items);

  return (
    <>
      <Section>
        <div className="toolbar">
          <div className="segmented">
            <button className={mode === "user" ? "active" : ""} onClick={() => setMode("user")}>
              按学习者
            </button>
            <button className={mode === "courses" ? "active" : ""} onClick={() => setMode("courses")}>
              按已学课程
            </button>
          </div>
          <label className="topn">
            推荐数量
            <input type="number" min="3" max="30" value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
          </label>
        </div>
        {mode === "user" ? (
          <label className="wide-label">
            选择学习者ID
            <select value={userId} onChange={(e) => setUserId(e.target.value)}>
              {users.map((user) => (
                <option key={user} value={user}>
                  {user}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <CoursePicker courses={courses} selectedCourses={selectedCourses} setSelectedCourses={setSelectedCourses} />
        )}
        {mode === "courses" && selectedCourses.length === 0 && <div className="inline-note compact">请先选择一门课程。</div>}
        {error && <div className="error-block compact">{error}</div>}
      </Section>
      <div className="chart-grid">
        <Section title="推荐分数" icon={BarChart3}>
          <div className="recommendation-chart-wrap">
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={chartItems} layout="vertical" margin={{ left: 80, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis dataKey="course_name" type="category" width={120} tick={{ fontSize: 12 }} />
                <Tooltip
                  content={
                    <RecommendationTooltip
                      pinnedCourse={pinnedCourse}
                      selectedCourses={selectedCourseRows}
                      onAnalyzeCourse={(course) => onAnalyzeCourse(course, selectedCourseRows)}
                    />
                  }
                  wrapperStyle={{ pointerEvents: "auto" }}
                />
                <Bar dataKey="recommendation_score" fill="#2563eb" radius={[0, 4, 4, 0]}>
                  {chartItems.map((entry) => (
                    <Cell
                      key={entry.course_id || entry.course_name}
                      cursor="pointer"
                      onClick={() => setPinnedCourse(entry)}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {pinnedCourse && (
              <div className="pinned-tooltip">
                <RecommendationTooltipCard
                  course={pinnedCourse}
                  selectedCourses={selectedCourseRows}
                  onAnalyzeCourse={(course) => onAnalyzeCourse(course, selectedCourseRows)}
                  onClose={() => setPinnedCourse(null)}
                  pinned
                />
              </div>
            )}
          </div>
        </Section>
        <Section title="推荐课程类别" icon={BookOpen}>
          <div className="pie-with-legend">
            <ResponsiveContainer width="100%" height={360}>
              <PieChart margin={{ top: 16, right: 84, bottom: 16, left: 84 }}>
                <Pie
                  data={recommendationCategories}
                  dataKey="count"
                  nameKey="category"
                  outerRadius={104}
                  innerRadius={54}
                  label={(props) => <PieLabel {...props} />}
                  labelLine
                >
                  {recommendationCategories.map((_, index) => (
                    <Cell key={index} fill={colorAt(index)} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <CategoryLegend rows={recommendationCategories} valueKey="count" />
          </div>
        </Section>
      </div>
      <DataTable rows={items} />
    </>
  );
}

function RecommendationTooltip({ active, payload, pinnedCourse, selectedCourses, onAnalyzeCourse }) {
  if (pinnedCourse || !active || !payload?.length) return null;
  const course = payload[0].payload;
  return <RecommendationTooltipCard course={course} selectedCourses={selectedCourses} onAnalyzeCourse={onAnalyzeCourse} />;
}

function RecommendationTooltipCard({ course, selectedCourses, onAnalyzeCourse, onClose, pinned = false }) {
  return (
    <div className={`chart-tooltip${pinned ? " pinned" : ""}`}>
      <div className="tooltip-title-row">
        <strong>{course.course_name || course.course_id}</strong>
        {onClose && (
          <button type="button" className="tooltip-close" onClick={onClose} aria-label="关闭固定悬浮窗">
            <X size={15} />
          </button>
        )}
      </div>
      <span>推荐分数：{formatScore(course.recommendation_score)}</span>
      {course.category && <small>{course.category}</small>}
      <button type="button" className="tooltip-action" onClick={() => onAnalyzeCourse(course)}>
        <MessageSquareText size={15} />
        了解更多该课程
      </button>
      {selectedCourses?.length > 0 && <small>将结合已选 {selectedCourses.length} 门课程分析</small>}
    </div>
  );
}

function CoursePicker({ courses, selectedCourses, setSelectedCourses }) {
  const [query, setQuery] = useState("");
  const courseById = useMemo(() => new Map(courses.map((course) => [course.course_id, course])), [courses]);
  const selectedSet = useMemo(() => new Set(selectedCourses), [selectedCourses]);
  const selectedCourseRows = useMemo(() => selectedCourses.map((id) => courseById.get(id)).filter(Boolean), [selectedCourses, courseById]);
  const matches = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return courses.filter((course) => !keyword || courseLabel(course).toLowerCase().includes(keyword));
  }, [courses, query]);
  const visible = matches;

  const resultText = query.trim()
    ? `匹配 ${formatNumber(matches.length)} 门课程，已全部显示`
    : `共 ${formatNumber(courses.length)} 门可选课程，已全部显示`;

  function toggle(id) {
    setSelectedCourses((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));
  }

  function removeCourse(id) {
    setSelectedCourses((current) => current.filter((item) => item !== id));
  }

  return (
    <div>
      <div className="selected-courses">
        <div className="selected-courses-head">
          <span>已选课程 {formatNumber(selectedCourseRows.length)}</span>
          {selectedCourseRows.length > 0 && (
            <button type="button" className="text-action" onClick={() => setSelectedCourses([])}>
              清空
            </button>
          )}
        </div>
        {selectedCourseRows.length > 0 ? (
          <div className="selected-chip-list">
            {selectedCourseRows.map((course) => (
              <span className="selected-chip" key={course.course_id}>
                <span>{courseLabel(course)}</span>
                <button type="button" onClick={() => removeCourse(course.course_id)} aria-label={`移除 ${course.course_name || course.course_id}`}>
                  <X size={14} />
                </button>
              </span>
            ))}
          </div>
        ) : (
          <div className="selected-empty">未选择课程</div>
        )}
      </div>
      <label className="search-box">
        <Search size={16} />
        <input placeholder="搜索课程名称、类别或 ID" value={query} onChange={(e) => setQuery(e.target.value)} />
      </label>
      <div className="result-count">{resultText}</div>
      <div className="course-list">
        {visible.map((course) => (
          <label key={course.course_id} className="course-item">
            <input type="checkbox" checked={selectedSet.has(course.course_id)} onChange={() => toggle(course.course_id)} />
            <span>{course.course_name || course.course_id}</span>
            <small>{course.category}</small>
          </label>
        ))}
      </div>
    </div>
  );
}

function courseLabel(course) {
  return `${course.course_name || course.course_id} [${course.category || "未分类"} | ${course.course_id}]`;
}

function categoryRows(items) {
  const map = new Map();
  items.forEach((item) => {
    const key = item.category || "未分类";
    map.set(key, (map.get(key) || 0) + 1);
  });
  return Array.from(map.entries()).map(([category, count]) => ({ category, count }));
}

function Analysis() {
  const { data, loading, error } = useApi("/api/analysis", []);
  if (loading) return <LoadingBlock />;
  if (error) return <ErrorBlock message={error} />;

  return (
    <>
      <div className="metrics-grid">
        <Metric label="推荐覆盖课程" value={`${data.summary.recommended_course_count} / ${data.summary.total_course_count}`} tone="blue" />
        <Metric label="覆盖率" value={formatPercent(data.latest.coverage)} tone="green" />
        <Metric label="热门推荐占比" value={formatPercent(data.latest.popular_recommendation_ratio)} tone="orange" />
        <Metric label="单门推荐平均出现" value={`${data.summary.avg_recommend_count} 次`} tone="purple" />
      </div>
      <Section title="参数组合指标对比" icon={Activity}>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={[...data.history].reverse()}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="trained_at" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="rmse" stroke="#2563eb" strokeWidth={2} />
            <Line type="monotone" dataKey="coverage" stroke="#16a34a" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Section>
      <div className="chart-grid">
        <Section title="推荐结果中出现最多的课程" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={[...data.hot_recommendations].reverse()} layout="vertical" margin={{ left: 80 }}>
              <XAxis type="number" />
              <YAxis dataKey="course_name" type="category" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="size" fill="#7c3aed" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
        <Section title="原始交互热门课程" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={[...data.popular_courses].reverse()} layout="vertical" margin={{ left: 80 }}>
              <XAxis type="number" />
              <YAxis dataKey="course_name" type="category" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="learn_count" fill="#f97316" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
      </div>
    </>
  );
}

function buildCourseAnalysisPrompt(course, selectedCourses = []) {
  const selectedText = selectedCourses.length
    ? selectedCourses
        .map((item, index) => `${index + 1}. ${item.course_name || item.course_id}（${item.category || "未分类"}）`)
        .join("\n")
    : "当前是按学习者推荐或未选择课程，请重点依据推荐课程简介与推荐分数进行解释。";

  return [
    `请分析推荐课程《${course.course_name || course.course_id}》。`,
    "",
    `推荐分数：${formatScore(course.recommendation_score)}`,
    `课程类别：${course.category || "未分类"}`,
    course.teacher ? `教师/来源：${course.teacher}` : "",
    "",
    "已选择/已学习课程：",
    selectedText,
    "",
    "请用 Markdown 输出，包含：",
    "1. 这门课主要讲什么",
    "2. 为什么它和我已经选择的课程有关",
    "3. 为什么这个推荐分数值得关注",
    "4. 学完后可能获得的能力或视野",
    "",
    "请给出正面、有说服力的推荐理由，但不要夸大。"
  ]
    .filter(Boolean)
    .join("\n");
}

function AiAnalysis({ pendingPrompt, clearPendingPrompt }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "## AI 课程分析\n\n你可以直接询问某门课程讲什么、适合什么学习目标，或者从推荐图表点击“了解更多该课程”生成推荐解释。",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const lastPromptKeyRef = useRef("");

  async function sendMessage(content, context = null) {
    const text = content.trim();
    if (!text || sending) return;
    const userMessage = { role: "user", content: text };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setSending(true);
    setError("");

    try {
      const response = await apiPost("/api/ai/chat", {
        messages: nextMessages
          .filter((message) => message.role === "user" || message.role === "assistant")
          .slice(-12),
        context,
      });
      setMessages((current) => [...current, { role: "assistant", content: response.content || "暂无回复。" }]);
    } catch (err) {
      setError(err.message);
      setMessages((current) => [...current, { role: "assistant", content: `**调用失败：** ${err.message}` }]);
    } finally {
      setSending(false);
    }
  }

  useEffect(() => {
    if (!pendingPrompt) return;
    const key = `${pendingPrompt.context?.recommended_course_id || ""}-${pendingPrompt.prompt}`;
    if (lastPromptKeyRef.current === key) return;
    lastPromptKeyRef.current = key;
    sendMessage(pendingPrompt.prompt, pendingPrompt.context);
    clearPendingPrompt();
  }, [pendingPrompt]);

  function submit(event) {
    event.preventDefault();
    sendMessage(input);
  }

  return (
    <div className="ai-shell">
      <div className="ai-chat">
        {messages.map((message, index) => (
          <div key={index} className={`chat-message ${message.role}`}>
            <div className="avatar">{message.role === "assistant" ? <Bot size={18} /> : <GraduationCap size={18} />}</div>
            <div className="message-bubble">
              <MarkdownMessage content={message.content} />
            </div>
          </div>
        ))}
        {sending && (
          <div className="chat-message assistant">
            <div className="avatar"><Bot size={18} /></div>
            <div className="message-bubble pending">
              <Loader2 className="spin" size={18} />
              正在分析课程内容与推荐依据
            </div>
          </div>
        )}
      </div>
      {error && <div className="error-block compact">{error}</div>}
      <form className="chat-input" onSubmit={submit}>
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="询问 AI：例如“数据挖掘：理论与算法适合什么基础的同学？”"
          rows={3}
        />
        <button className="primary" type="submit" disabled={sending || !input.trim()}>
          {sending ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
          发送
        </button>
      </form>
    </div>
  );
}

function DataTable({ rows }) {
  if (!rows || rows.length === 0) return <div className="state-block">暂无数据</div>;
  const columns = Object.keys(rows[0]).slice(0, 8);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{labelForColumn(col)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 50).map((row, index) => (
            <tr key={index}>
              {columns.map((col) => (
                <td key={col}>{String(row[col] ?? "-")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [active, setActive] = useState("overview");
  const [visitedTabs, setVisitedTabs] = useState(() => new Set(["overview"]));
  const [pendingAiPrompt, setPendingAiPrompt] = useState(null);

  function showTab(id) {
    setActive(id);
    setVisitedTabs((currentTabs) => {
      if (currentTabs.has(id)) return currentTabs;
      return new Set([...currentTabs, id]);
    });
  }

  function handleAnalyzeCourse(course, selectedCourses = []) {
    const prompt = buildCourseAnalysisPrompt(course, selectedCourses);
    setPendingAiPrompt({
      prompt,
      context: {
        recommended_course_id: course.course_id,
        recommended_course_name: course.course_name,
        recommendation_score: course.recommendation_score,
        selected_course_ids: selectedCourses.map((item) => item.course_id),
      },
    });
    showTab("ai");
  }

  const tabs = [
    { id: "overview", label: "数据概览", icon: BarChart3, component: <Overview /> },
    { id: "training", label: "模型训练", icon: RefreshCw, component: <Training /> },
    { id: "recommend", label: "个性化推荐", icon: Sparkles, component: <Recommendations onAnalyzeCourse={handleAnalyzeCourse} /> },
    { id: "analysis", label: "结果分析", icon: Activity, component: <Analysis /> },
    {
      id: "ai",
      label: "AI分析",
      icon: MessageSquareText,
      component: <AiAnalysis pendingPrompt={pendingAiPrompt} clearPendingPrompt={() => setPendingAiPrompt(null)} />,
    },
  ];
  const current = tabs.find((tab) => tab.id === active) || tabs[0];

  return (
    <div className="app">
      <aside>
        <div className="brand">
          <GraduationCap size={28} />
          <div>
            <strong>基于 Spark ALS 的课程协同过滤推荐系统</strong>
          </div>
        </div>
        <nav>
          {tabs.map((tab) => (
            <button key={tab.id} className={active === tab.id ? "active" : ""} onClick={() => showTab(tab.id)}>
              <tab.icon size={18} />
              {tab.label}
            </button>
          ))}
        </nav>
      </aside>
      <main>
        <header>
          <div>
            <h1>{current.label}</h1>

          </div>
        </header>
        {tabs.map((tab) =>
          visitedTabs.has(tab.id) ? (
            <div key={tab.id} className={active === tab.id ? "page-panel active" : "page-panel"} aria-hidden={active !== tab.id}>
              {tab.component}
            </div>
          ) : null,
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
