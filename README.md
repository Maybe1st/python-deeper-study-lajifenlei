# 基于深度学习的垃圾分类系统的设计与实现

## 项目简介

本系统是一个基于 Flask Web 框架与 PyTorch 深度学习模型的垃圾分类智能识别平台。系统面向垃圾图像分类场景，能够对上传图片中的玻璃、纸张、纸板、塑料、金属、其他垃圾等类别进行识别，并在后台保存识别记录与统计结果。

系统使用 TrashNet 公开垃圾分类数据集进行模型训练，管理员可手动发起模型训练，训练完成后自动加载新权重。启动服务不会自动训练。

---

## API 接口

所有接口返回 JSON，格式 `{ "code": 0, "data": ... }` 或 `{ "code": 1, "message": "..." }`。数据均来自数据库或模型真实推理，无 mock。

| 接口 | 方法 | 权限 | 说明 |
| --- | --- | --- | --- |
| `/api/dashboard` | GET | 登录 | 数据概览统计：识别次数、置信度、类别/类型分布、置信度区间、各类平均置信度、数据集分布、识别趋势；管理员可查看混淆矩阵 |
| `/api/dataset` | GET | 登录 | TrashNet 数据集各类别真实样本数 |
| `/api/records` | GET | 登录 | 识别记录分页，`page`/`per_page` 参数 |
| `/api/records/<id>` | DELETE | 登录 | 删除识别记录 |
| `/api/classify` | POST | 登录 | 上传图片分类，返回模型真实推理结果 |
| `/api/model/status` | GET | 登录 | 模型是否存在及最近训练指标 |
| `/api/training/records` | GET | 管理员 | 训练历史记录 |
| `/api/training/start` | POST | 管理员 | 发起训练，body: `{epochs, batch_size, learning_rate}` |
| `/api/users` | GET | 管理员 | 用户列表 |
| `/api/users/<id>` | DELETE | 管理员 | 删除用户 |

**`/api/dashboard` 主要数据字段（均为真实统计，无 mock）：**

| 字段 | 说明 |
| --- | --- |
| `class_distribution` | 识别记录按六类统计 |
| `category_distribution` | 可回收物 / 干垃圾统计 |
| `confidence_bins` | 置信度区间分布（0–50%、50–70%…） |
| `class_avg_confidence` | 各类平均置信度与次数 |
| `dataset_distribution` | TrashNet 目录真实样本数 |
| `daily_trend` | 近 14 日识别趋势 |
| `confusion_matrix` | 管理员：最近一次训练测试集混淆矩阵 |

**`/api/classify` 返回字段说明（数值由模型实时推理产生）：**

```json
{
  "code": 0,
  "data": {
    "predicted_class": "plastic",
    "predicted_cn": "塑料",
    "category": "可回收物",
    "confidence": 0.8734,
    "top3": [
      {
        "class_name": "plastic",
        "class_cn": "塑料",
        "category": "可回收物",
        "confidence": 0.8734
      },
      {
        "class_name": "glass",
        "class_cn": "玻璃",
        "category": "可回收物",
        "confidence": 0.0621
      },
      {
        "class_name": "paper",
        "class_cn": "纸张",
        "category": "可回收物",
        "confidence": 0.0312
      }
    ]
  }
}
```

---

## 功能模块

| 模块 | 功能说明 |
| --- | --- |
| **用户认证** | 用户注册、登录、退出；基于 Session 的会话管理；管理员与普通用户两级角色权限控制 |
| **数据概览** | 核心指标卡片 + ECharts 图表：垃圾类别分布、可回收/干垃圾分布、置信度区间分布、各类平均置信度、TrashNet 样本分布、近 14 日识别趋势 |
| **垃圾识别** | 支持点击上传单张图片，实时执行模型推理，返回预测类别、垃圾类型、置信度与 Top-3 结果，并自动保存识别记录 |
| **识别记录** | 识别记录分页浏览，查看历史识别结果，删除记录，导出 CSV（中文表头） |
| **数据集信息** | 展示 TrashNet 数据集统计（各类别样本数、数据集路径、下载地址、模型架构说明） |
| **模型训练** | 管理员可配置训练轮次、批次大小、学习率并启动在线训练；左侧模型指标卡与右侧准确率曲线同高展示，另含损失曲线与训练历史记录 |
| **用户管理** | 管理员查看所有用户列表，删除普通用户账户 |

---

## 垃圾类别

系统支持识别以下六类垃圾：

| 类别 ID | 英文名称 | 中文名称 | 垃圾类型 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | glass | 玻璃 | 可回收物 | 玻璃瓶、玻璃制品等 |
| 1 | paper | 纸张 | 可回收物 | 报纸、书本、纸箱纸等 |
| 2 | cardboard | 纸板 | 可回收物 | 纸箱、硬纸板等 |
| 3 | plastic | 塑料 | 可回收物 | 塑料瓶、塑料袋等 |
| 4 | metal | 金属 | 可回收物 | 易拉罐、金属制品等 |
| 5 | trash | 其他垃圾 | 干垃圾 | 不可回收的其他垃圾 |

---

## 标准版目录结构

```text
python-deeper-study-lajifenlei/
├── app.py                         # Flask 应用工厂
├── run.py                         # 启动脚本（初始化数据库、启动服务，无模型时提示训练）
├── train.py                       # 独立模型训练脚本
├── config.py                      # 系统配置（端口、数据库路径、模型路径、类别映射、训练参数）
├── database.py                    # SQLite 数据库初始化与连接管理
├── requirements.txt               # Python 依赖包清单
├── mac_run.sh                     # macOS/Linux 一键启动脚本
├── window_run.bat                 # Windows 一键启动脚本
├── 项目说明.md                    # 项目功能与目录结构说明
├── 训练手册.md                    # 模型训练与数据集使用手册
├── DATASET.md                     # 数据集下载地址与引用说明
├── ml/
│   ├── __init__.py
│   └── classifier.py              # ResNet18 / TrashCNN 模型定义、训练与推理
├── services/
│   ├── auth_service.py            # 用户认证业务逻辑
│   ├── classify_service.py        # 图片分类与记录管理
│   ├── stats_service.py           # 数据统计与数据集信息
│   └── train_service.py           # 训练记录管理
├── routes/
│   ├── auth.py                    # 登录、注册、退出路由
│   ├── main.py                    # 页面路由
│   └── api.py                     # API 接口路由
├── templates/                     # Jinja2 HTML 模板
│   ├── base.html                  # 基础模板（侧边栏、本地资源引用）
│   ├── login.html                 # 登录页
│   ├── register.html              # 注册页
│   ├── dashboard.html             # 数据概览
│   ├── classify.html              # 垃圾识别页
│   ├── history.html               # 识别记录页
│   ├── dataset.html               # 数据集信息页
│   ├── train.html                 # 模型训练页
│   └── users.html                 # 用户管理页
├── static/
│   ├── css/
│   │   └── app.css                # 自定义样式（绿色环保主题）
│   ├── uploads/                   # 用户上传图片存储目录
│   └── vendor/                    # 本地前端资源（无需 CDN）
│       ├── bootstrap/             # Bootstrap 5.3 CSS/JS
│       └── echarts/               # ECharts 5.5 图表库
├── data/
│   └── app.db                     # SQLite 数据库文件（自动生成）
├── models_weights/                # 训练输出的模型权重
│   ├── trash_classifier.pth       # 分类模型权重
│   └── model_meta.json            # 模型元信息（准确率、训练时间等）
├── trashnet-master/               # TrashNet 数据集
│   └── data/
│       └── dataset-resized/       # 已缩放数据集（6 类，2527 张）
│           ├── glass/
│           ├── paper/
│           ├── cardboard/
│           ├── plastic/
│           ├── metal/
│           └── trash/
├── 运行步骤必看/
│   ├── window.md                  # Windows 运行步骤说明
│   └── mac.md                     # macOS 运行步骤说明
└── 用户上传检测文件/              # 测试用垃圾图片样例
    ├── 玻璃.jpg
    ├── 纸张.jpg
    ├── 纸板.jpg
    ├── 塑料.jpg
    ├── 金属.jpg
    ├── 其他垃圾.jpg
    └── 使用说明.md
```

---

## 技术栈

| 类别 | 技术 |
| --- | --- |
| **后端框架** | Python 3.12 + Flask 3.0 |
| **用户认证** | Flask Session + Werkzeug 密码哈希 |
| **数据库** | SQLite 3 |
| **深度学习** | PyTorch + TorchVision（ResNet18 迁移学习，兼容 TrashCNN） |
| **图像处理** | Pillow |
| **数据处理** | NumPy |
| **前端** | HTML5 + Bootstrap 5.3 + ECharts 5.5 |
| **数据集来源** | TrashNet（Stanford CS 229 公开数据集） |

---

## 环境要求

| 项目 | 要求 |
| --- | --- |
| **Python** | 3.12 |
| **操作系统** | Windows / macOS / Linux |
| **内存** | 建议 4GB 以上（模型推理与训练时使用） |
| **磁盘** | 约 1GB（含依赖、数据集与模型权重） |
| **GPU** | 可选，支持 CUDA / CPU 自动回退 |
| **浏览器** | Chrome / Firefox / Safari / Edge 现代版本 |

---

## 账号信息

系统初始化时自动创建以下默认账户：

| 角色 | 用户名 | 密码 | 权限 |
| --- | --- | --- | --- |
| 管理员 | admin | admin123 | 全部功能，含模型训练、用户管理 |
| 普通用户 | 自行注册 | 自行设置 | 垃圾识别、识别记录、数据概览、数据集查看 |

新注册用户默认为普通用户角色，仅能访问自身识别数据；管理员可查看全部用户数据。

---

## 获取完整源码

- 网站：[AI源码](https://www.aiyuanma.vip)
- 本项目详情：[https://www.aiyuanma.vip/posts/python-deeper-study-lajifenlei](https://www.aiyuanma.vip/posts/python-deeper-study-lajifenlei)

## 联系方式

- QQ：861077046
- 邮箱：861077046@qq.com

> 本文由 AI源码 自动同步，完整源码与技术支持请访问官网。
