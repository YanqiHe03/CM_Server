# RunPod Serverless 部署指南

## 📁 文件结构

```
CM_Server/
├── handler.py          # RunPod Serverless 入口文件
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
└── DEPLOY_RUNPOD.md    # 本部署指南
```

## 🚀 部署步骤

### 方法一：使用 GitHub 集成（推荐）✨

这是最简单的部署方式，RunPod 会自动从 GitHub 拉取代码并构建镜像。

#### 步骤 1：准备 GitHub 仓库

确保你的仓库包含以下文件：

```
your-repo/
├── handler.py          # 必须
├── requirements.txt    # 必须  
├── Dockerfile          # 必须
└── ...
```

#### 步骤 2：推送代码到 GitHub

```bash
git init
git add handler.py requirements.txt Dockerfile
git commit -m "Add RunPod Serverless handler"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

#### 步骤 3：在 RunPod 连接 GitHub

1. 登录 [RunPod Console](https://runpod.io/console/serverless)
2. 点击左侧菜单 **"Serverless"**
3. 点击 **"+ New Endpoint"**
4. 选择 **"GitHub Repo"** 选项卡

#### 步骤 4：授权 GitHub 访问

1. 点击 **"Connect GitHub"** 按钮
2. 在弹出窗口中授权 RunPod 访问你的 GitHub
3. 选择要部署的仓库

#### 步骤 5：配置构建设置

| 配置项 | 建议值 | 说明 |
|--------|--------|------|
| **Repository** | 选择你的仓库 | - |
| **Branch** | `main` | 监听的分支 |
| **Dockerfile Path** | `Dockerfile` | Dockerfile 相对路径 |
| **Auto Build** | ✅ 开启 | 推送代码自动重新构建 |

#### 步骤 6：配置 Endpoint 设置

| 配置项 | 建议值 | 说明 |
|--------|--------|------|
| **GPU Type** | RTX 3080/3090/4080 | Qwen-VL 2B 需要 6GB+ 显存 |
| **Max Workers** | 1-3 | 根据并发需求 |
| **Idle Timeout** | 5-10 秒 | 空闲多久后关闭 worker |
| **Flash Boot** | ✅ 开启 | 减少冷启动时间 |

#### 步骤 7：设置环境变量

在 Endpoint 配置页面的 **"Environment Variables"** 部分添加：

| 变量名 | 值 | 说明 |
|--------|-----|------|
| `HF_MODEL_ID` | `your-username/cm-gallery-vlm` | 你的 HF 模型仓库路径 |
| `HF_TOKEN` | `hf_xxxxx` | HF API Token（私有仓库需要） |

#### 步骤 8：创建 Endpoint

1. 点击 **"Create Endpoint"**
2. 等待 RunPod 拉取代码并构建镜像（首次约 5-10 分钟）
3. 构建完成后，你会看到 Endpoint ID

#### 🔄 自动更新

开启 Auto Build 后，每次你 `git push` 到 main 分支：
- RunPod 自动检测变更
- 自动重新构建镜像
- 自动部署新版本

---

### 方法二：使用 Docker Hub

如果你偏好手动管理镜像，可以使用这个方法。

#### 1. 构建 Docker 镜像

```bash
# 在 CM_Server 目录下
docker build -t your-dockerhub-username/cm-gallery-vlm:latest .
```

#### 2. 推送到 Docker Hub

```bash
docker login
docker push your-dockerhub-username/cm-gallery-vlm:latest
```

#### 3. 在 RunPod 创建 Serverless Endpoint

1. 登录 [RunPod Console](https://runpod.io/console/serverless)
2. 点击 **"+ New Endpoint"**
3. 选择 **"Docker Image"** 选项卡
4. 填写配置：
   - **Container Image**: `your-dockerhub-username/cm-gallery-vlm:latest`
   - **GPU Type**: 选择合适的GPU（建议 RTX 3080/3090 或更高）
   - **Max Workers**: 根据需求设置（建议 1-3）
   - **Idle Timeout**: 建议 5-10 秒
   - **Flash Boot**: 开启（减少冷启动时间）

#### 4. 设置环境变量

| 变量名 | 值 | 说明 |
|--------|-----|------|
| `HF_MODEL_ID` | `your-username/cm-gallery-vlm` | 你的 HF 模型仓库路径 |
| `HF_TOKEN` | `hf_xxxxx` | HF API Token（私有仓库需要） |

---

## 📤 API 调用示例

### Endpoint URL 格式

```
https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
```

### Python 调用示例

```python
import requests
import base64

# RunPod API 配置
RUNPOD_API_KEY = "your_runpod_api_key"
ENDPOINT_ID = "your_endpoint_id"

def call_compliment_api(image_path=None, image_url=None, user_text=None):
    """
    调用 Complimentary Machine API
    
    Args:
        image_path: 本地图片路径（与 image_url 二选一）
        image_url: 图片 URL（与 image_path 二选一）
        user_text: 可选的用户文本
    """
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建 input
    input_data = {}
    
    if image_path:
        # 读取本地图片并转为 base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        input_data["image"] = image_base64
    elif image_url:
        input_data["image"] = image_url
    else:
        raise ValueError("必须提供 image_path 或 image_url")
    
    if user_text:
        input_data["user_text"] = user_text
    
    payload = {"input": input_data}
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


# 使用示例
if __name__ == "__main__":
    # 方式1：使用本地图片
    result = call_compliment_api(
        image_path="test_image.jpg",
        user_text="这是我画的画"
    )
    print(result)
    
    # 方式2：使用图片 URL
    result = call_compliment_api(
        image_url="https://example.com/image.jpg"
    )
    print(result)
```

### cURL 调用示例

```bash
# 使用图片 URL
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer {RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/image.jpg",
      "user_text": "这是我的作品"
    }
  }'
```

### 异步调用（长时间任务）

```python
import requests
import time

def call_async(input_data):
    # 提交任务
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    response = requests.post(url, json={"input": input_data}, headers=headers)
    job_id = response.json()["id"]
    
    # 轮询结果
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    while True:
        status = requests.get(status_url, headers=headers).json()
        if status["status"] == "COMPLETED":
            return status["output"]
        elif status["status"] == "FAILED":
            raise Exception(status.get("error"))
        time.sleep(1)
```

---

## 📋 API 输入/输出格式

### 输入格式

```json
{
  "input": {
    "image": "<base64_string 或 URL>",  // 必填
    "user_text": "可选的用户文本"        // 可选
  }
}
```

### 输出格式

**成功响应：**
```json
{
  "delayTime": 123,
  "executionTime": 456,
  "id": "job-id",
  "output": {
    "compliment": "生成的赞美文本"
  },
  "status": "COMPLETED"
}
```

**错误响应：**
```json
{
  "output": {
    "error": "错误信息"
  },
  "status": "COMPLETED"
}
```

---

## ⚡ 性能优化建议

### 1. 预烘焙模型（减少冷启动时间）

在 Dockerfile 中取消注释以下部分：

```dockerfile
ARG HF_MODEL_ID=your-username/cm-gallery-vlm
ARG HF_TOKEN
RUN python -c "from transformers import AutoModelForVision2Seq, AutoProcessor; \
    AutoModelForVision2Seq.from_pretrained('${HF_MODEL_ID}', trust_remote_code=True, token='${HF_TOKEN}'); \
    AutoProcessor.from_pretrained('${HF_MODEL_ID}', trust_remote_code=True, token='${HF_TOKEN}')"
```

构建时传入参数：
```bash
docker build \
  --build-arg HF_MODEL_ID=your-username/cm-gallery-vlm \
  --build-arg HF_TOKEN=hf_xxxxx \
  -t your-dockerhub-username/cm-gallery-vlm:latest .
```

### 2. 使用 Network Volume

在 RunPod 中创建 Network Volume，用于缓存模型：
- 设置 `HF_HOME=/runpod-volume/huggingface`
- 首次启动后模型会被缓存，后续冷启动更快

### 3. Flash Boot

在 Endpoint 设置中开启 Flash Boot，可以显著减少冷启动时间。

---

## 🔧 调试技巧

### 本地测试

```python
# test_local.py
import handler

# 模拟加载模型
handler.load_model()

# 模拟 RunPod job
test_job = {
    "input": {
        "image": "https://example.com/test.jpg",
        "user_text": "测试"
    }
}

result = handler.handler(test_job)
print(result)
```

### 查看 RunPod 日志

在 RunPod Console 中：
1. 进入 Endpoint 详情页
2. 点击 "Logs" 标签
3. 查看 worker 日志和请求日志

---

## 💰 成本估算

- **冷启动**：约 30-60 秒（取决于是否预烘焙模型）
- **推理时间**：约 1-3 秒/请求
- **GPU 选择建议**：
  - Qwen-VL 2B 模型约需 4-6GB 显存
  - 推荐：RTX 3080 (10GB) 或 RTX 4080 (16GB)
  - 最低：RTX 3060 (12GB)

---

## ❓ 常见问题

### Q: 冷启动时间太长？
A: 使用预烘焙模型方式构建镜像，或开启 Flash Boot。

### Q: 模型加载失败？
A: 检查 `HF_MODEL_ID` 是否正确，私有仓库需要设置 `HF_TOKEN`。

### Q: 显存不足？
A: 选择更大显存的 GPU 类型，或考虑使用量化版本的模型。

### Q: 并发处理？
A: RunPod Serverless 会自动扩展 workers，每个 worker 独立处理一个请求。
