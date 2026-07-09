import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BookOpen,
  GraduationCap,
  Loader2,
  RefreshCw,
  Search,
  Sparkles,
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

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const COLORS = ["#2563eb", "#16a34a", "#f97316", "#7c3aed", "#0891b2", "#db2777", "#64748b", "#ca8a04"];

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
      <div className="section-head">
        <div>
          <h2>
            <Icon size={20} />
            {title}
          </h2>
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

function Overview() {
  const { data, loading, error } = useApi("/api/overview", []);
  if (loading) return <LoadingBlock />;
  if (error) return <ErrorBlock message={error} />;

  const summary = data.summary || {};
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
            <BarChart data={[...data.popular_courses].reverse()} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" />
              <YAxis dataKey="course_name" type="category" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="learn_count" radius={[0, 4, 4, 0]}>
                {[...data.popular_courses].reverse().map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>
        <Section title="课程类别交互分布" icon={Activity}>
          <ResponsiveContainer width="100%" height={340}>
            <PieChart>
              <Pie
                data={data.category_distribution}
                dataKey="interaction_count"
                nameKey="category"
                outerRadius={116}
                innerRadius={58}
                label
              >
                {data.category_distribution.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Section>
      </div>
      <div className="chart-grid">
        <Section title="用户选课数量分布" icon={Activity}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.user_course_histogram}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
        <Section title="课程被学习次数分布" icon={Activity}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.course_learn_histogram}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#16a34a" radius={[4, 4, 0, 0]} />
            </BarChart>
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
            rank
            <input value={form.rank} type="number" min="2" max="50" onChange={(e) => setForm({ ...form, rank: Number(e.target.value) })} />
          </label>
          <label>
            maxIter
            <input value={form.max_iter} type="number" min="3" max="30" onChange={(e) => setForm({ ...form, max_iter: Number(e.target.value) })} />
          </label>
          <label>
            regParam
            <input value={form.reg_param} type="number" step="0.01" min="0.001" max="1" onChange={(e) => setForm({ ...form, reg_param: Number(e.target.value) })} />
          </label>
          <label>
            topN
            <input value={form.top_n} type="number" min="3" max="30" onChange={(e) => setForm({ ...form, top_n: Number(e.target.value) })} />
          </label>
        </div>
        {message && <div className="inline-note">{message}</div>}
      </Section>
      <div className="metrics-grid">
        <Metric label="RMSE" value={latest.rmse ?? "-"} tone="blue" />
        <Metric label="覆盖率" value={formatPercent(latest.coverage)} tone="green" />
        <Metric label="多样性" value={latest.diversity ?? "-"} tone="orange" />
        <Metric label="热门推荐占比" value={formatPercent(latest.popular_recommendation_ratio)} tone="purple" />
      </div>
      <DataTable rows={data.history} />
    </>
  );
}

function Recommendations() {
  const usersState = useApi("/api/recommendations/users", []);
  const coursesState = useApi("/api/courses?limit=800", []);
  const [mode, setMode] = useState("user");
  const [userId, setUserId] = useState("");
  const [selectedCourses, setSelectedCourses] = useState([]);
  const [topN, setTopN] = useState(8);
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");
  const users = usersState.data?.users || [];
  const courses = coursesState.data?.courses || [];

  useEffect(() => {
    if (!userId && users.length) setUserId(users[0]);
  }, [users, userId]);

  useEffect(() => {
    if (!selectedCourses.length && courses.length) setSelectedCourses(courses.slice(0, 3).map((course) => course.course_id));
  }, [courses, selectedCourses.length]);

  async function loadRecommendations() {
    setError("");
    try {
      const path =
        mode === "user"
          ? `/api/recommendations/by-user?user_id=${encodeURIComponent(userId)}&top_n=${topN}`
          : `/api/recommendations/by-courses?${selectedCourses.map((id) => `course_ids=${encodeURIComponent(id)}`).join("&")}&top_n=${topN}`;
      const data = await apiGet(path);
      setItems(data.items || []);
    } catch (err) {
      setError(err.message);
      setItems([]);
    }
  }

  useEffect(() => {
    if ((mode === "user" && userId) || (mode === "courses" && selectedCourses.length)) {
      loadRecommendations();
    }
  }, [mode, userId, selectedCourses.join(","), topN]);

  if (usersState.loading || coursesState.loading) return <LoadingBlock />;
  if (usersState.error) return <ErrorBlock message={usersState.error} />;

  return (
    <>
      <Section title="个性化推荐" icon={Sparkles}>
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
            Top-N
            <input type="number" min="3" max="30" value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
          </label>
        </div>
        {mode === "user" ? (
          <label className="wide-label">
            选择学习者 user_id
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
        {error && <div className="error-block compact">{error}</div>}
      </Section>
      <div className="chart-grid">
        <Section title="推荐分数" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={[...items].reverse()} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" />
              <YAxis dataKey="course_name" type="category" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="recommendation_score" fill="#2563eb" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
        <Section title="推荐课程类别" icon={BookOpen}>
          <ResponsiveContainer width="100%" height={360}>
            <PieChart>
              <Pie data={categoryRows(items)} dataKey="count" nameKey="category" outerRadius={115} innerRadius={56} label>
                {categoryRows(items).map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Section>
      </div>
      <DataTable rows={items} />
    </>
  );
}

function CoursePicker({ courses, selectedCourses, setSelectedCourses }) {
  const [query, setQuery] = useState("");
  const visible = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return courses
      .filter((course) => !keyword || `${course.course_name}${course.course_id}${course.category}`.toLowerCase().includes(keyword))
      .slice(0, 80);
  }, [courses, query]);

  function toggle(id) {
    setSelectedCourses((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(0, 8)));
  }

  return (
    <div>
      <label className="search-box">
        <Search size={16} />
        <input placeholder="搜索课程名称、类别或 ID" value={query} onChange={(e) => setQuery(e.target.value)} />
      </label>
      <div className="course-list">
        {visible.map((course) => (
          <label key={course.course_id} className="course-item">
            <input type="checkbox" checked={selectedCourses.includes(course.course_id)} onChange={() => toggle(course.course_id)} />
            <span>{course.course_name || course.course_id}</span>
            <small>{course.category}</small>
          </label>
        ))}
      </div>
    </div>
  );
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

function DataTable({ rows }) {
  if (!rows || rows.length === 0) return <div className="state-block">暂无数据</div>;
  const columns = Object.keys(rows[0]).slice(0, 8);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
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
  const tabs = [
    { id: "overview", label: "数据概览", icon: BarChart3, component: <Overview /> },
    { id: "training", label: "模型训练", icon: RefreshCw, component: <Training /> },
    { id: "recommend", label: "个性化推荐", icon: Sparkles, component: <Recommendations /> },
    { id: "analysis", label: "结果分析", icon: Activity, component: <Analysis /> },
  ];
  const [active, setActive] = useState("overview");
  const current = tabs.find((tab) => tab.id === active) || tabs[0];

  return (
    <div className="app">
      <aside>
        <div className="brand">
          <GraduationCap size={28} />
          <div>
            <strong>慕课课程推荐</strong>
            <span>Netlify 前端展示版</span>
          </div>
        </div>
        <nav>
          {tabs.map((tab) => (
            <button key={tab.id} className={active === tab.id ? "active" : ""} onClick={() => setActive(tab.id)}>
              <tab.icon size={18} />
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="api-box">
          <span>API</span>
          <code>{API_BASE_URL || "/api"}</code>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <h1>{current.label}</h1>
            <p>基于 Spark ALS 的课程协同过滤推荐系统</p>
          </div>
        </header>
        {current.component}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
