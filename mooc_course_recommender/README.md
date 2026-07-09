# 基于 Spark 的慕课课程协同过滤推荐系统设计与实现

## 项目简介

本项目面向在线教育/慕课平台场景，使用 Apache Spark 和 Spark MLlib ALS 算法实现一个简易课程个性化推荐系统。系统支持慕课用户、课程和交互数据接入，完成清洗预处理、统计分析、协同过滤训练、推荐结果存储，并通过 Streamlit 提供可交互的可视化界面。

当前默认 `data/raw/` 已替换为 THU-KEG/MOOCCubeX 官方数据子集，适合课程设计演示和答辩。完整 MOOCCubeX 数据规模较大，本项目抽取部分用户选课记录和课程元数据构造可本地运行的协同过滤推荐样本。

## 技术栈

- Python：项目主要开发语言
- PySpark：大数据处理、清洗、统计和模型训练
- Spark MLlib ALS：隐式反馈协同过滤推荐
- Streamlit：Web 可视化交互界面
- Pandas：读取 Spark 输出结果用于页面展示
- Plotly：交互式统计图表
- Parquet/CSV：中间数据和推荐结果存储

## 目录结构

```text
mooc_course_recommender/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ output/
├─ src/
│  ├─ config.py
│  ├─ generate_sample_data.py
│  ├─ load_data.py
│  ├─ preprocess.py
│  ├─ train_als.py
│  ├─ recommend.py
│  ├─ evaluate.py
│  └─ storage.py
├─ app.py
├─ requirements.txt
├─ README.md
└─ run_pipeline.py
```

## 数据说明

原始数据位于 `data/raw/`，支持 CSV 或 JSON。当前数据由 MOOCCubeX 的 `course.json` 和 `user.json` 用户样本转换得到，包含：

- `users.csv`：`user_id`, `course_order`
- `courses.csv`：`course_id`, `course_name`, `category`, `teacher`, `description`
- `interactions.csv`：`user_id`, `course_id`, `enroll`, `watch_ratio`, `exercise_ratio`, `preference`

如果没有显式评分，系统使用学习行为构造隐式偏好分数：

```text
preference = 选课基础分 + 视频观看贡献 + 练习完成贡献
```

处理结果保存在 `data/processed/`：

- `interactions_clean`
- `user_index_map`
- `course_index_map`
- `als_train_data`
- `popular_courses`
- `category_distribution`
- `user_course_counts`
- `course_learn_counts`

推荐输出保存在 `data/output/`：

- `recommendations`
- `latest_metrics`
- `metrics_history`

所有核心结果同时保存为 Parquet 和 CSV 目录，便于 Spark 读取和人工查看。

## 环境安装

建议在已创建的虚拟环境中执行：

```bash
pip install -r requirements.txt
```

PySpark 需要本机已安装 Java，并正确配置 `JAVA_HOME`。

Windows 环境中 Spark 可能提示 `Did not find winutils.exe`。本项目在保存本地结果时使用 Spark 计算、Pandas 写入 Parquet/CSV 的兼容方式，避免该警告影响课程设计演示流程。如果需要使用 Spark 原生 Hadoop 文件提交器写本地 Parquet，可另行配置 `HADOOP_HOME` 和 `winutils.exe`。

## 运行步骤

1. 生成示例数据：

```bash
python src/generate_sample_data.py
```

如果需要重新下载并转换 MOOCCubeX 官方数据子集，可执行：

```bash
python src/prepare_mooccubex_data.py --max-users 3000 --max-interactions 30000
```

该脚本会从 THU-KEG/MOOCCubeX 官方下载链接读取 `course.json` 和 `user.json` 的用户样本，转换为本项目统一的 `users.csv`、`courses.csv` 和 `interactions.csv`。

2. 执行完整数据处理与训练流程：

```bash
python run_pipeline.py
```

3. 启动可视化系统：

```bash
streamlit run app.py
```

也可以使用新增的前后端分离展示方式：`frontend/` 部署到 Netlify，`backend/` 在本机运行 FastAPI，并通过 ngrok/cpolar 暴露公网 HTTPS 地址。详细步骤见 `NETLIFY_FRONTEND_GUIDE.md`。

## 系统功能说明

### 数据接入

`src/load_data.py` 使用 Spark 读取 `data/raw/` 下的用户、课程和交互数据，支持 CSV/JSON 文件。

### 数据清洗与预处理

`src/preprocess.py` 完成缺失值过滤、重复用户-课程交互合并、低交互用户/课程过滤，并通过 `StringIndexer` 将字符串 `user_id` 和 `course_id` 编码为 ALS 可用的整数索引。

### 统计分析

Spark 生成用户数、课程数、交互数、热门课程 Top 10、课程类别分布、用户选课数量分布和课程被学习次数分布等统计表。

### 智能化推荐

`src/train_als.py` 使用 Spark MLlib ALS，设置 `implicitPrefs=True`，基于隐式反馈偏好矩阵训练协同过滤模型。支持调节：

- `rank`
- `maxIter`
- `regParam`
- `topN`

模型训练后为所有用户批量生成 Top-N 课程推荐，并关联课程名称、类别、教师和推荐分数。

个性化推荐页面提供两种交互方式：

- 按学习者 `user_id` 推荐：使用 ALS 为指定历史用户生成 Top-N 个性化推荐。
- 按已学课程推荐：用户选择自己之前学过的课程，系统从 Spark 清洗后的交互矩阵中寻找共同学习过这些课程的相似学习者，再统计这些相似学习者还学习过的课程，排除已选课程后按共现强度、相似用户数和平均偏好分数生成推荐。这种方式适合没有固定 `user_id` 的演示或冷启动场景。

### 模型评估

系统计算并保存：

- RMSE：基于随机划分测试集的预测误差
- 推荐覆盖率：推荐结果覆盖的不同课程比例
- 推荐多样性：单个用户推荐列表中的平均类别数量
- 热门推荐占比：推荐结果是否过度集中于少数课程

### 可视化交互

`app.py` 使用 Streamlit 实现四个页面：

- 数据概览：展示 Spark 处理后的数据源统计图表
- 模型训练：提供 ALS 参数控件，点击后调用 PySpark 训练并写入新结果
- 个性化推荐：支持选择学习者 `user_id`，也支持选择已学课程后基于共现协同过滤生成推荐
- 结果分析：展示不同参数组合指标对比、热门课程与推荐课程对比、应用案例说明

页面图表读取 `data/processed/` 和 `data/output/` 的 Spark 输出结果，不使用手工写死图表数据。

## 与课程设计任务书要求的对应关系

1. 数据采集与接入：导入慕课用户、课程和交互数据，也提供模拟数据生成脚本。
2. 数据清洗与预处理：完成去重、缺失处理、低频过滤、ID 编码和用户-课程偏好矩阵构造。
3. 智能化分析与处理：使用 Spark MLlib ALS 协同过滤算法完成个性化课程推荐。
4. 存储：将清洗数据、统计结果、推荐结果和模型评估指标保存为 Parquet/CSV。
5. 可视化：使用 Streamlit 展示数据源统计、算法参数调整、推荐结果和分析图表。
6. 应用案例：输入或选择某个学习者 ID，或选择自己之前学过的课程，系统根据历史学习行为推荐适合继续学习的慕课课程。

## 答辩时可讲的技术亮点

- 使用 Spark 完成从数据接入、清洗、统计到 ALS 训练的完整大数据处理流程。
- 将非显式评分的学习行为转化为隐式偏好分数，更符合慕课平台实际业务。
- ALS 使用 `implicitPrefs=True`，适合选课、观看、练习等隐式反馈推荐场景。
- 训练参数可在 Web 页面交互式调整，训练结果会重新写入 `data/output/` 并刷新展示。
- 同时提供 RMSE、覆盖率、多样性和热门推荐占比，避免只用单一误差指标评价推荐系统。
- 推荐结果关联课程元数据，便于解释推荐内容和进行类别分布分析。
