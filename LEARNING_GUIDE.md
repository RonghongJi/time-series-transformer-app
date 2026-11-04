# 从本地 Python 程序到公网 Transformer 网站：全流程学习笔记

> 项目地址：<https://github.com/RonghongJi/time-series-transformer-app>
>
> 公网应用：<https://ronghongji-time-series-transformer.streamlit.app/>

---

# ⭐ 先补这些知识点

这次项目表面上是在“做一个网站”，实际串联了 Python、机器学习、Web 服务、网络、Git 和云部署。建议按下面的优先级补课。

## 第一优先级：必须理解

### 1. Python 基础

- 变量、列表、字典、循环和条件判断
- 函数：`def`、参数、返回值
- 模块导入：`import`
- 类与对象：`class`、`__init__`、方法
- 异常与报错信息的阅读方法
- 文件和项目目录结构

本项目对应代码：模拟时序、制作数据集、定义 Transformer、执行训练和生成表格。

### 2. Python 环境与依赖管理

- Python 解释器是什么
- Anaconda 的 `base` 环境是什么
- `venv` 虚拟环境是什么
- `pip` 把包安装到哪里
- 为什么同一台电脑会有多个 Python
- `which python`、`python -m pip` 和 `python -m streamlit` 的意义
- `requirements.txt` 如何复现运行环境

本项目最典型的错误：

```text
ModuleNotFoundError: No module named 'torch'
```

不是没有安装 PyTorch，而是 PyTorch 安装在 `.venv`，网页却被 Anaconda `base` 中的 Streamlit 启动。以后优先使用：

```bash
python -m pip install 包名
python -m streamlit run app.py
```

这样可以保证安装依赖和运行程序使用同一个 Python。

### 3. Web 服务基础

- 浏览器不是直接执行 Python，而是向 Web 服务发送请求
- Streamlit 把 Python 程序包装成 Web 应用
- 前端、后端、服务器分别是什么
- HTTP、HTTPS、URL、IP、域名和端口的区别
- `localhost`、`127.0.0.1`、`0.0.0.0` 的区别
- 本地访问、局域网访问和公网访问的区别

核心结构：

```text
浏览器 → HTTP请求 → Streamlit服务 → Python代码 → 网页结果
```

### 4. Git 和 GitHub

- Git 是本地版本管理工具
- GitHub 是远程代码托管平台
- 仓库、提交、分支、远程地址是什么
- `git init`、`git add`、`git commit`、`git push`
- `.gitignore` 为什么不能少
- 公开仓库不能存放密码、Token 和公司数据

### 5. 云部署

- 本地程序为什么关机后就不能访问
- 云端如何重新安装依赖并运行程序
- GitHub 仓库如何成为部署来源
- 域名、HTTPS、反向代理和云端容器的基本概念
- 本机 Apple GPU 与云端 CPU 环境的差别

## 第二优先级：理解项目原理

### 6. 时序数据

- 时间顺序为什么不能随便打乱
- 滑动窗口：用过去 `N` 个点预测下一个点
- 训练集和测试集为什么要按时间切分
- 收益率、价格、趋势、周期和随机噪声
- 分类问题与回归问题的区别

当前项目做的是二分类：

```text
过去24个时间点 → Transformer → 预测下一时刻上涨或下跌
```

### 7. Transformer 基础

- Tensor（张量）和数据形状
- `batch × sequence × feature`
- 输入映射 `Linear`
- 位置编码
- Self-Attention
- Transformer Encoder
- 分类头
- Logits、Softmax 和置信度

当前输入形状大致为：

```text
[批量样本数, 时间窗口长度, 每个时间点的特征数]
```

例如：

```text
[32, 24, 1]
```

表示一次送入 32 条样本，每条样本包含过去 24 个时间点，每个点有 1 个特征。

### 8. 神经网络训练

- Epoch、Batch、Loss、Accuracy
- 前向传播和反向传播
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`
- 训练模式 `model.train()`
- 推理模式 `model.eval()`
- `torch.no_grad()`
- 过拟合与测试集

## 第三优先级：准备做更正式的产品

### 9. 安全与权限

- API Key 不能写进源码
- GitHub Secrets 和 Streamlit Secrets
- 登录认证和访问控制
- 限制输入规模、运行时间和并发
- 公司数据的授权、脱敏和审计
- 公开接口不等于获得调用授权

### 10. 后续扩展

- FastAPI：构建更通用的后端 API
- WebSocket/SSE：更灵活的实时推送
- Docker：统一开发与部署环境
- 数据库：保存用户、任务和历史结果
- 队列：处理耗时训练任务
- Nginx/Caddy：反向代理和 HTTPS
- CI/CD：自动测试与部署

---

# 🚀 明天新项目：一次成功的标准操作顺序

下面是推荐的唯一主流程。按顺序执行，不要同时使用“网页上传文件”和“本地 Git 推送”两条路线。

> 核心原则：GitHub 网页只创建**完全空白的仓库**，所有代码都从本地第一次推送上去。

## A. 本地完成代码并测试

```bash
mkdir -p ~/new-streamlit-demo
cd ~/new-streamlit-demo

python3 -m venv .venv
source .venv/bin/activate

python -m pip install streamlit

# 创建并写好 app.py 后，本地测试
python -m streamlit run app.py
```

确认网页正常后，按 `Control + C` 停止服务。

## B. 在第一次 git add 之前准备文件

项目至少应有：

```text
new-streamlit-demo/
├── app.py
├── requirements.txt
└── .gitignore
```

`requirements.txt` 只列程序直接需要的包和版本。可用下面的命令查询版本：

```bash
python -c "import streamlit; print(streamlit.__version__)"
```

`.gitignore` 提前写好：

```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/secrets.toml
.agents/
.claude/
```

先检查目录，确认没有密码、Token、公司数据或不该公开的文件。

## C. 先配置提交身份，再创建第一次提交

```bash
git config --global user.name "Ronghong Ji"
git config --global user.email "jironghong1998@gmail.com"

git init -b main

# 明确写出要添加的文件，不要直接使用 git add .
git add app.py requirements.txt .gitignore

git status
git commit -m "Create Streamlit app"
```

`git status` 中不应该出现 `.venv`、`.agents`、`.claude`、Secrets 或其他无关文件。

## D. GitHub 网页只创建空仓库

在 <https://github.com/new> 创建仓库：

- 填写一个从未使用过的仓库名；
- 选择 `Public` 或 `Private`；
- **不要**勾选 `Add a README file`；
- `.gitignore` 选择 `None`；
- License 选择 `None`；
- 创建后不要在网页上传或编辑任何文件。

这样远程仓库没有第一次提交，不会与本地形成两套历史。

## E. 连接空仓库并第一次推送

将下面的地址换成新仓库地址：

```bash
git remote add origin https://github.com/RonghongJi/新仓库名.git
git remote -v
git push -u origin main
```

成功后，本地 `main` 会跟踪远程 `origin/main`。以后不再需要 `-u origin main`，直接 `git push` 即可。

## F. 部署 Streamlit Community Cloud

在 <https://share.streamlit.io> 创建应用：

```text
Repository: RonghongJi/新仓库名
Branch: main
Main file path: app.py
Python: 与本地兼容的版本
```

部署完成后获得固定的 `https://...streamlit.app` 地址。

## G. 以后每次更新网站

```bash
cd ~/new-streamlit-demo
source .venv/bin/activate

# 修改后先在本地测试
python -m streamlit run app.py
```

确认后按 `Control + C`，然后：

```bash
git status
git diff

# 只添加本次确实要提交的文件
git add app.py
git commit -m "Describe this update"
git push
```

GitHub 收到新提交后，Streamlit Community Cloud 会自动重新部署。

## H. 明天不要做的事情

- 不要在本地提交后，又在空 GitHub 仓库中通过网页上传同一批文件；
- 不要勾选 GitHub 的 README、`.gitignore` 或 License 初始化选项；
- 不要在没看 `git status` 时直接执行 `git add .`；
- 不要上传 `.venv`、Secrets、Token 或公司数据；
- 不要用 `git push --force` 解决普通的同步问题；
- 不要混用多个 Python，启动时使用 `python -m streamlit run app.py`。

---

# 一、最终做成了什么

本项目完成了一条完整链路：

```text
模拟时序数据
    ↓
制作滑动窗口样本
    ↓
训练 PyTorch Transformer
    ↓
实时显示训练损失和准确率
    ↓
在测试集上预测上涨/下跌
    ↓
用表格标记预测正确或错误
    ↓
通过 Streamlit 生成网页
    ↓
代码上传 GitHub
    ↓
Streamlit Community Cloud 部署
    ↓
获得全球可访问的 HTTPS 地址
```

---

# 二、准备本地 Python 环境

## 1. 检查 Python

```bash
python3 --version
which python3
```

本机结果：

```text
Python 3.12.4
/opt/anaconda3/bin/python3
```

这说明使用的是 Anaconda 提供的 Python。

## 2. 创建项目目录

```bash
mkdir -p ~/web-demo
cd ~/web-demo
pwd
ls -la
```

## 3. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

激活后终端通常出现：

```text
(.venv)
```

检查解释器：

```bash
which python
python --version
```

正确路径应类似：

```text
/Users/rhji/web-demo/.venv/bin/python
```

## 4. 安装依赖

```bash
python -m pip install streamlit
python -m pip install torch
```

检查 PyTorch 和 Apple GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

本机返回 `True`，表示 PyTorch 可以使用 Apple 的 MPS GPU 加速。

---

# 三、建立第一个本地网页

## 1. Streamlit 的作用

普通 Python 程序主要在终端输入和输出。Streamlit 提供输入框、按钮、图表、表格等网页组件，并负责启动 Web 服务。

示例：

```python
import streamlit as st

st.title("我的第一个本地网页")
name = st.text_input("请输入你的名字")

if st.button("打招呼"):
    st.success(f"你好，{name}！")
```

## 2. 启动网页

推荐命令：

```bash
python -m streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

停止服务：

```text
Control + C
```

---

# 四、理解 localhost 和 8501

地址：

```text
http://localhost:8501
```

可以拆成：

| 部分 | 含义 |
|---|---|
| `http` | 浏览器与服务通信的协议 |
| `localhost` | 当前这台电脑，通常指向 `127.0.0.1` |
| `8501` | Streamlit 默认监听的端口 |

`8501` 不是我们手动配置的端口映射，而是 Streamlit 的默认端口。可以主动修改：

```bash
python -m streamlit run app.py --server.port 9000
```

这时访问：

```text
http://localhost:9000
```

## 监听所有网卡

```bash
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

`0.0.0.0` 表示服务接受来自所有网络接口的连接，但它不是浏览器中要输入的访问地址。

如果只想让同一局域网中的其他设备访问，需要使用 Mac 的局域网 IP，例如：

```text
http://192.168.1.23:8501
```

这仍然不是公网网站。

---

# 五、时序 Transformer 的工作原理

## 1. 生成模拟数据

程序生成包含三部分的收益率：

```text
当前收益率 = 动量 + 周期变化 + 随机噪声
```

其中：

- 动量让相邻时间点具有一定相关性；
- 周期项制造规律性波动；
- 噪声让问题不至于完全确定。

再把收益率累积为类似价格的时序曲线。

## 2. 滑动窗口

如果窗口长度为 24：

```text
第 1～24 个点 → 预测第 25 个点的方向
第 2～25 个点 → 预测第 26 个点的方向
第 3～26 个点 → 预测第 27 个点的方向
```

标签定义：

```text
下一时刻值 > 当前值 → 1（上涨）
下一时刻值 ≤ 当前值 → 0（下跌）
```

## 3. Transformer 模型结构

```text
单变量时序
    ↓
Linear：把1维特征映射为32维
    ↓
加入位置编码
    ↓
Transformer Encoder
    ↓
取最后一个时间位置的表示
    ↓
Linear分类头
    ↓
上涨/下跌两个Logits
```

位置编码很重要，因为 Transformer 的注意力机制本身不知道哪个点在前、哪个点在后。

## 4. 训练循环

每个 Batch 的关键步骤：

```python
optimizer.zero_grad()
logits = model(batch_x)
loss = loss_function(logits, batch_y)
loss.backward()
optimizer.step()
```

含义：

1. 清空上一次梯度；
2. 模型生成预测；
3. 计算预测与真实标签的差距；
4. 反向传播计算梯度；
5. 优化器更新模型参数。

## 5. 测试与结果表格

测试时使用：

```python
model.eval()

with torch.no_grad():
    ...
```

测试结果包括：

- 当前值；
- 下一时刻真实值；
- 实际上涨或下跌；
- 模型预测方向；
- 置信度；
- `✅ 正确` 或 `❌ 错误`。

注意：模拟数据上的准确率不能代表模型在真实金融、工业或业务数据上的效果。

---

# 六、网页为什么能实时变化

训练过程中，每完成一个 Epoch，代码都会更新：

- `st.progress()`：进度条；
- `st.empty()`：可被新内容覆盖的区域；
- `st.line_chart()`：训练损失曲线；
- `st.code()`：最近的训练日志；
- `st.metric()`：准确率等指标。

这里的“实时”是训练循环主动更新网页组件。对于更复杂的多用户系统，通常会使用 WebSocket、SSE、任务队列和后台 Worker。

---

# 七、排错：为什么已经安装 torch 仍然报错

报错：

```text
ModuleNotFoundError: No module named 'torch'
```

Traceback 中出现：

```text
/opt/anaconda3/lib/python3.12/site-packages/streamlit/
```

但 PyTorch 安装在：

```text
/Users/rhji/web-demo/.venv/
```

说明当时实际启动的是 Anaconda `base` 中的 Streamlit，而不是 `.venv` 中的 Streamlit。

解决方法：

```bash
python -m streamlit run app.py
```

排错命令：

```bash
which python
python -c "import torch; print(torch.__version__)"
python -c "import streamlit; print(streamlit.__file__)"
```

经验法则：不要只看终端有没有 `(.venv)`，还要用 `which python` 和模块路径确认真正使用的解释器。

---

# 八、准备 GitHub 仓库

## 1. 依赖文件

`requirements.txt`：

```text
streamlit==1.66.1
torch==2.15.0
numpy==2.5.4
pandas==3.1.0
```

云端读取它并安装依赖。

## 2. 忽略文件

`.gitignore`：

```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/secrets.toml
.agents/
.claude/
```

`.venv` 不应上传：它很大、依赖本机平台，而且云端会根据 `requirements.txt` 创建自己的环境。

## 3. 先配置身份，再初始化本地 Git

```bash
git config --global user.name "Ronghong Ji"
git config --global user.email "jironghong1998@gmail.com"

git init -b main

# 明确添加需要的文件，不使用 git add .
git add app.py requirements.txt .gitignore
git status
git commit -m "Create time series Transformer app"
```

提交身份由本机 `git config` 决定，不会自动继承浏览器中的 GitHub 登录身份。应在第一次提交前配置，否则 Git 可能根据电脑用户名生成错误的本地邮箱。

## 4. 在 GitHub 创建完全空白的仓库

推荐使用 GitHub 网页创建仓库，但不要初始化任何内容：

- 不添加 README；
- 不选择 `.gitignore` 模板；
- 不选择 License；
- 不通过网页上传文件。

创建后只复制 HTTPS 仓库地址，在本地执行：

```bash
git remote add origin https://github.com/RonghongJi/仓库名.git
git remote -v
git push -u origin main
```

这样远程第一次提交直接来自本地，不会产生不相关历史，也不需要合并。

## 5. GitHub CLI（可选，不是主流程）

使用 Homebrew 安装：

```bash
HOMEBREW_NO_AUTO_UPDATE=1 brew install gh
```

登录命令：

```bash
gh auth login --hostname github.com --git-protocol https --web
```

如果登录成功，也可以用 GitHub CLI 创建仓库。但初学阶段建议坚持上面的单一路线：网页只建空仓库，本地负责第一次推送。

## 6. 本次为什么曾经需要合并

本次实验中，本地先创建了提交，GitHub 网页又独立上传文件并创建了另一份提交。两边没有共同父提交，因此第一次推送被拒绝，必须使用 `--allow-unrelated-histories` 合并。

这属于踩坑复盘，不是推荐步骤。以后按照“空远程仓库 + 本地首次推送”流程，不需要执行该合并命令。

---

# 九、部署成公网网站

## 1. 为什么不能直接把 localhost 发给别人

`localhost` 在每台电脑上都表示“这台电脑自己”。别人打开 `localhost:8501`，访问的是对方自己的电脑，不是你的 Mac。

直接暴露本机还会遇到：

- 家用或公司网络没有公网入口；
- 路由器 NAT；
- 防火墙；
- 动态 IP；
- 没有 HTTPS；
- 电脑关机后服务消失；
- 安全风险。

## 2. Community Cloud 部署结构

```text
GitHub仓库
    ↓ 拉取源码
Streamlit Community Cloud
    ↓ 安装 requirements.txt
云端运行 app.py
    ↓
HTTPS公网域名
```

部署配置：

```text
Repository: RonghongJi/time-series-transformer-app
Branch: main
Main file path: app.py
Python: 3.12
```

公开地址：

```text
https://ronghongji-time-series-transformer.streamlit.app/
```

## 3. 公网为什么看不到 8501

本地开发时直接访问 Streamlit 的 `8501` 端口。

公网访问时：

```text
访客浏览器
    ↓ HTTPS 443
Streamlit云端入口/反向代理
    ↓ 内部转发
应用运行环境
```

云平台负责 HTTPS 证书、域名和内部端口转发，因此公开 URL 不需要写 `:8501`。

## 4. 本地 GPU 与云端 CPU

本机：

```text
torch.backends.mps.is_available() == True
```

Community Cloud 通常不是 Apple 设备，因此 MPS 不可用，程序会回退到 CPU。这也是为什么部署程序必须对不同运行环境做判断。

---

# 十、以后如何更新网站

推荐的标准流程：

```bash
cd ~/web-demo
source .venv/bin/activate

# 本地修改并测试
python -m streamlit run app.py

# 查看修改
git status
git diff

# 提交修改：只添加确实发生变化的文件
git add app.py
git commit -m "Describe the change"

# 推送到 GitHub
git push
```

GitHub 更新后，Streamlit Community Cloud 会检测变化并重新部署。

仓库建立后，建议统一在本地修改并使用 `git push`，不要再通过网页上传同一批文件，以免形成不必要的同步和冲突问题。

---

# 十一、公开网站的安全注意事项

当前网站允许任何访客触发训练。正式使用前需要考虑：

1. 限制时序长度、Epoch 和模型规模；
2. 限制同时运行的任务数量；
3. 设置运行超时；
4. 避免访问本地或公司敏感文件；
5. 不把 API Key 写进 GitHub；
6. 对上传文件检查类型和大小；
7. 需要时增加登录验证；
8. 明确真实业务数据的授权范围；
9. 记录异常和运行日志；
10. 不把演示模型误用为真实投资或业务决策依据。

---

# 十二、建议的补课顺序

## 第 1 周：Python 与环境

- Python 函数、类、列表、NumPy 数组
- 虚拟环境和 pip
- 读懂 Traceback
- 独立完成一个命令行小程序

## 第 2 周：Web 与网络

- HTTP 请求和响应
- IP、域名和端口
- localhost、局域网和公网
- Streamlit 常用组件
- 做一个带输入、图表和表格的小应用

## 第 3 周：PyTorch

- Tensor 形状
- Dataset 和 DataLoader
- Linear、Loss、Optimizer
- 训练循环和测试循环
- 做一个简单二分类模型

## 第 4 周：时序与 Transformer

- 滑动窗口
- 时间顺序切分训练集和测试集
- Attention 和位置编码
- Transformer Encoder
- 把分类改为数值回归

## 第 5 周：Git 和部署

- Git 的工作区、暂存区和提交
- GitHub 远程仓库
- requirements.txt
- 云部署、日志和排错
- Secrets 与最小权限

---

# 十三、建议继续完成的练习

1. 把模拟数据改成上传 CSV；
2. 允许用户选择目标列；
3. 对训练集做标准化，并只使用训练集统计量；
4. 增加验证集和 Early Stopping；
5. 把上涨/下跌分类改成预测下一个数值；
6. 增加真实值与预测值的对比曲线；
7. 保存模型并提供下载；
8. 添加任务运行时间和设备信息；
9. 给公开页面增加使用说明和免责声明；
10. 使用 Git 推送一次修改，观察云端自动重新部署。

---

# 十四、常用命令速查

```bash
# 进入项目
cd ~/web-demo

# 激活虚拟环境
source .venv/bin/activate

# 检查解释器
which python
python --version

# 安装依赖
python -m pip install -r requirements.txt

# 启动本地网站
python -m streamlit run app.py

# 指定端口
python -m streamlit run app.py --server.port 9000

# 查看 Git 状态
git status

# 查看代码变化
git diff

# 明确添加需要提交的文件
git add app.py
git commit -m "Update app"

# 推送代码
git push
```

---

# 十五、一句话总结

你这次不是单纯“写了一个网页”，而是完成了一个小型机器学习 Web 产品的完整生命周期：

> 管理 Python 环境 → 编写模型 → 实时可视化 → 本地提供 Web 服务 → 用 Git 管理代码 → 在云端部署 → 通过 HTTPS 向公网提供服务。

