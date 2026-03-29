# TCP 推理服务协议说明

## 0. 他人快速启动（已发布镜像）

镜像：
- `registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0`

步骤：
1. `docker login registry.cn-hangzhou.aliyuncs.com`
2. `docker pull registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0`
3. 使用 [docker-compose.tcp-infer.runtime.yaml](/C:/Users/xzjs/Downloads/piping_update/mmyolo/deploy/docker-compose.tcp-infer.runtime.yaml) + `.env` 启动
4. `docker compose -f docker-compose.tcp-infer.runtime.yaml up -d`
5. `docker logs -f mmyolo-tcp-infer`

## 1. 服务启动

### 1.1 本地 Docker Compose 启动（源码构建方式）

在仓库根目录执行：

```bash
docker compose -f deploy/docker-compose.tcp-infer.yaml up --build -d
```

如果是“只有镜像、没有源码”的部署方式，请使用 `docker-compose.tcp-infer.runtime.yaml`（见上方“他人快速启动”）。

如果你要把 Windows 本地目录映射进容器，建议先在仓库根目录放一个 `.env`：

```bash
LOCAL_INPUT_DIR=C:/Users/yourname/data/piping_inputs
PATH_MAPPINGS=C:/Users/yourname/data/piping_inputs=/data/local_input
```

这样 TCP 请求里传入 Windows 路径（如 `C:\\Users\\yourname\\data\\piping_inputs\\a.jpg`）时，服务会自动转为容器内路径 `/data/local_input/a.jpg`。

如果你没有 GPU，把 `deploy/docker-compose.tcp-infer.yaml` 里的 `MODEL_DEVICE` 改成 `cpu`，并删除 `gpus: all`。

查看日志：

```bash
docker logs -f mmyolo-tcp-infer
```

停止：

```bash
docker compose -f deploy/docker-compose.tcp-infer.yaml down
```

### 1.2 本地脚本启动

```bash
bash run_tcp_infer_server.sh
```

默认监听：`0.0.0.0:9000`

## 2. 请求格式（客户端 -> 服务端）

请求是一行 JSON（JSONL），最少包含 `file_path`：

```json
{
  "task_id": "task-001",
  "file_path": "C:\\Users\\yourname\\data\\piping_inputs\\demo.jpg",
  "config": "/workspace/mmyolo/projects/piping/configs/yolov8_s_fast_8xb16-500e_ours.py",
  "checkpoint": "/workspace/mmyolo/weights/best_coco_破裂_precision_epoch_25.pth",
  "device": "cuda:0",
  "score_thr": 0.3,
  "out_dir": "/workspace/mmyolo/output/tcp_results",
  "save_visualization": false,
  "infer_stride": 3,
  "max_frames": -1,
  "progress_every_frames": 30,
  "return_records": false
}
```

说明：
- `file_path` 必填，支持图片、图片目录、视频文件。
- `file_path` 支持 Windows 路径（如 `C:\\...`），服务会按 `PATH_MAPPINGS` 自动映射到容器路径。
- 如果不传 `config/checkpoint/device/score_thr`，服务使用启动时默认值。
- 视频场景下，`return_records=false` 可避免回包过大。

## 3. 响应格式（服务端 -> 客户端）

服务端会持续发送 JSONL：

- 接收成功：
```json
{"type":"ack","status":"accepted","task_id":"task-001"}
```

- 处理中（多条）：
```json
{"type":"progress","status":"running","task_id":"task-001","progress":35,"stage":"infer_image","message":"processing image 2/5"}
```

- 完成：
```json
{"type":"result","status":"ok","task_id":"task-001","result":{...}}
```

- 失败：
```json
{"type":"error","status":"error","task_id":"task-001","message":"...","traceback":"..."}
```

## 4. Python 客户端示例

```python
import json
import socket

request = {
    "task_id": "task-001",
    "file_path": "C:\\Users\\yourname\\data\\piping_inputs\\demo.jpg"
}

with socket.create_connection(("127.0.0.1", 9000), timeout=30) as sock:
    sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))

    buffer = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line:
                continue
            msg = json.loads(line.decode("utf-8"))
            print(msg)
            if msg.get("type") in {"result", "error"}:
                break
```
