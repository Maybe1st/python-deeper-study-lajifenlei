# 基于深度学习的垃圾分类系统

这是一个最小可运行版的垃圾分类 Web 平台实现，基于 Flask + PyTorch。功能涵盖用户认证、图片上传推理、识别记录管理、仪表盘统计、模型训练触发以及基础的用户管理界面。该实现目标是：从 README 的模板演化成一个可直接运行的原型，便于学习、调试与扩展。

---

## 主要特性
- 用户注册 / 登录 / 退出（Session）
- 管理员与普通用户权限
- 图片上传并调用 PyTorch 模型实时推理（ResNet18 占位实现）
- 识别记录保存（SQLite）与分页查询 / 删除
- 仪表盘统计接口（类别分布、可回收/干垃圾分布、置信度区间、各类平均置信度、数据集分布、近 14 日识别趋势）
- 管理员可发起训练任务，后台执行 train.py 并写入训练记录
- 简单前端页面（Jinja2 + Bootstrap + ECharts）：仪表盘、识别页、历史、用户管理、数据集、训练

---

## 技术栈
- 语言：Python 3.x
- 后端：Flask
- 深度学习：PyTorch + TorchVision（ResNet18 占位）
- 数据库：SQLite（data/app.db）
- 图像处理：Pillow
- 前端：Bootstrap + ECharts

---

## 快速开始
1. 克隆仓库并进入目录：

```bash
git clone https://github.com/Maybe1st/python-deeper-study-lajifenlei.git
cd python-deeper-study-lajifenlei
```

2. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. 启动服务（开发模式）：

```bash
python run.py
```

服务默认监听 0.0.0.0:5000（开发模式）。

默认管理员账号：
- 用户名：`admin`
- 密码：`admin123`

首次启动会自动初始化 SQLite 数据库并创建管理员账号。

---

## 目录结构（实现版）
```
python-deeper-study-lajifenlei/
├── app.py                 # Flask 应用工厂，加载模型并注册蓝图
├── run.py                 # 启动脚本
├── train.py               # 训练脚本（可在后台由 /api/training/start 触发）
├── config.py              # 配置：路径、类别映射、常量
├── database.py            # SQLite 初始化与简易访问
├── requirements.txt       # 依赖清单
├── templates/             # Jinja2 模板（dashboard, classify, history, users...）
├── static/                # 静态资源（css、uploads）
├── ml/                    # 模型封装：ml/classifier.py（ResNet18 占位实现）
├── routes/                # 路由蓝图（auth、main、api）
├── data/                  # 运行时数据（app.db）
├── models_weights/        # 训练输出模型/占位权重（trash_classifier.pth）
└── trashnet-master/       # 可选：将 TrashNet 数据集放在这里以运行训练
```

---

## 主要 API（JSON 返回，格式：{ "code":0, "data":... } 或 { "code":1, "message":... }）
- POST /api/classify
  - 说明：上传图片进行分类（需登录）
  - 表单字段：file（multipart/form-data）
  - 返回示例（data）：predicted_class, predicted_cn, category, confidence, top3

- GET /api/model/status
  - 说明：检查模型权重是否存在
  - 返回 data: {"exists": true/false}

- POST /api/training/start
  - 说明：管理员触发训练（后台运行 train.py），body 为 JSON：{"epochs":.., "batch_size":.., "learning_rate":..}
  - 返回 data 包含 training_id

- GET /api/training/records
  - 说明：管理员查询训练历史（started_at, finished_at, status, accuracy）

- GET /api/records?page=1&per_page=10
  - 说明：分页查询当前用户（管理员可查看全部）的识别记录

- DELETE /api/records/<id>
  - 说明：删除识别记录（管理员可删除任意记录）

- GET /api/dashboard
  - 说明：返回系统统计数据（类别分布、置信度分布、每类平均置信度、数据集样本分布、近 14 日趋势）

- GET /api/users (管理员)
  - 说明：列出用户（id, username, is_admin）

- DELETE /api/users/<id> (管理员)
  - 说明：删除用户（保护最后一个管理员）

---

## 模型与训练
- 模型代码位于 ml/classifier.py，当前实现使用 torchvision.models.resnet18（未加载预训练权重），并将全连接层替换为 6 类输出。
- 如果你已有训练好的权重文件（models_weights/trash_classifier.pth），放入该路径，应用启动时会尝试加载它用于推理。
- 训练脚本：train.py。数据集格式使用 torchvision.datasets.ImageFolder，默认数据目录为 `trashnet-master/data/dataset-resized`（按 README 的 TrashNet 目录组织，6 个子目录：glass, paper, cardboard, plastic, metal, trash）。
- 训练会将权重保存到 `models_weights/trash_classifier.pth`，并写入同路径下的 meta JSON（例如 `models_weights/trash_classifier.json`）包含 accuracy 信息，后台触发训练会把此 accuracy 写入 training_records 表。

---

## 数据库（SQLite）结构简要
- users(id, username, password_hash, is_admin)
- records(id, user_id, filename, predicted_class, confidence, created_at)
- training_records(id, started_at, finished_at, status, accuracy)

首次启动会创建这些表并插入默认管理员（admin/admin123）。

---

## 开发与调试提示
- 若要在 GPU 上训练或推理，请在 config.py 中将 DEVICE 改为 'cuda'，并确保安装的 PyTorch 支持 CUDA。
- 日志与异常可在控制台查看（app.run debug=True）。生产部署请使用 gunicorn / uWSGI 并关闭 debug。
- 若 /api/classify 返回置信度均为 0 或预测结果不合理，可能是模型权重为空或未训练；可以通过 `python train.py` 在本机先训练一个快速示例（数据集需存在）。
- 上传文件保存在 static/uploads 下；可定期清理以节省磁盘空间。

---

## 已知限制与后续改进
- 当前模型使用 ResNet18 占位实现，没有集成更复杂的数据增强、验证集评估、早停、学习率调度等训练工具。
- 权限控制简单：基于 session 的 is_admin 标志；生产中建议使用更严格的认证、CSRF 防护、输入校验与文件类型验证。
- 前端为教学示例，样式与交互可进一步完善（例如：异步加载、分页、导出 CSV、文件大小限制、进度显示）。

---

## 贡献与许可证
欢迎提交 issue 或 PR 改进功能。如果需要我把代码打包成 zip 并发给你，或把项目进一步完善（例如在 CI 中自动化训练、添加单元测试、Docker 化），告诉我具体需求我来完成。

---

如果你希望我把 README 调整成英文版、添加更详细的 API 文档（含请求/响应示例）、或直接在仓库生成 Release 包（zip），告诉我你的偏好，我接着更新。