# Netlify 前端 + 本地 FastAPI 后端

本项目保留原来的 `app.py` Streamlit 展示方式，同时新增了一套前后端分离展示方式：

```text
访问者浏览器 -> Netlify 静态前端 -> ngrok/cpolar 公网地址 -> 本机 FastAPI 后端 -> data/processed 与 data/output
```

## 本地启动后端

先安装接口依赖：

```powershell
pip install -r requirements-api.txt
```

启动 FastAPI 后端：

```powershell
cd "D:\desktop\人工智能与大数据平台实训\mooc_course_recommender"
$env:CORS_ALLOW_ORIGINS="http://localhost:5173"
.\scripts\run-backend-dev.ps1
```

本地接口地址：

```text
http://localhost:8000/api/health
```

## 本地启动前端

```powershell
cd "D:\desktop\人工智能与大数据平台实训\mooc_course_recommender\frontend"
npm install
npm run dev
```

本地前端地址：

```text
http://localhost:5173
```

## 临时公网后端

FastAPI 默认端口是 `8000`。另开 PowerShell：

```powershell
ngrok http 8000
```

如果 ngrok 在国内打不开，换 cpolar/natapp，把本机 `8000` 映射成 HTTPS 公网地址。

拿到公网地址后，例如：

```text
https://xxxx.ngrok-free.dev
```

重新启动后端，并允许 Netlify 域名跨域：

```powershell
cd "D:\desktop\人工智能与大数据平台实训\mooc_course_recommender"
$env:CORS_ALLOW_ORIGINS="https://你的站点名.netlify.app"
.\scripts\run-backend-dev.ps1
```

## Netlify 配置

仓库根目录已有 `netlify.toml`：

```toml
[build]
base = "frontend"
publish = "dist"
command = "npm run build"
```

在 Netlify 项目的 Environment variables 里添加：

```text
VITE_API_BASE_URL=https://你的后端公网地址
```

然后重新部署。

## 注意

- Netlify 只托管 `frontend` 静态页面。
- 本机后端、Java/Spark 环境、ngrok/cpolar 隧道必须保持运行。
- 如果隧道地址变了，需要更新 Netlify 环境变量 `VITE_API_BASE_URL` 并重新部署。
- 原 Streamlit 仍然可以用：`streamlit run app.py`。
