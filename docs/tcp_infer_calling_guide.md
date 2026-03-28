# TCP 推理调用文档

## 1. 能力说明

TCP 服务接收一条任务请求（JSONL），读取文件并分析，过程中持续回传进度，完成后回传最终结果。

支持输入类型：
- 单张图片
- 图片目录（目录下所有图片）
- 单个视频文件

## 2. 通信协议

- 传输协议：TCP
- 编码：UTF-8
- 消息格式：**一行一个 JSON**（JSONL）
- 交互方式：客户端发 1 条请求，服务端回多条响应（`ack` / `progress` / `result` 或 `error`）

## 3. 请求输入（Client -> Server）

### 3.1 最小请求

```json
{
  "file_path": "C:\\Users\\yourname\\data\\piping_inputs\\demo.jpg"
}
```

### 3.2 完整请求字段

```json
{
  "task_id": "task-001",
  "file_path": "C:\\Users\\yourname\\data\\piping_inputs\\demo.jpg",
  "base_dir": "/workspace/mmyolo",
  "config": "/workspace/mmyolo/projects/piping/configs/yolov8_n_fast_8xb16-500e_defect_lower_lr_ag.py",
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

### 3.3 字段定义

- `file_path` (string, 必填)  
  输入文件路径。支持 Windows 路径（`C:\...`）和 Linux 路径。
- `task_id` (string, 可选)  
  任务 ID；不传会自动生成 UUID。
- `base_dir` (string, 可选)  
  相对路径解析基准目录；默认仓库根目录。
- `config` (string, 可选)  
  模型配置路径；不传用服务默认值。
- `checkpoint` (string, 可选)  
  模型权重路径；不传用服务默认值。
- `device` (string, 可选)  
  推理设备，如 `cuda:0` / `cpu`。
- `score_thr` (number, 可选)  
  置信度阈值。
- `out_dir` (string, 可选)  
  输出目录（jsonl 和可视化文件）。
- `save_visualization` (boolean, 可选，图片任务有效)  
  是否保存可视化图片。
- `infer_stride` (int, 可选，视频任务有效)  
  每隔多少帧做一次推理。
- `max_frames` (int, 可选，视频任务有效)  
  最多处理帧数，`-1` 表示全部。
- `progress_every_frames` (int, 可选，视频任务有效)  
  每处理多少帧回传一次进度。
- `return_records` (boolean, 可选，视频任务建议 false)  
  是否把逐帧明细放到最终响应里。视频长时建议 `false`，避免回包过大。

## 4. 路径映射（Windows -> 容器 Linux）

当 `file_path` 是 Windows 路径时，服务会按 `PATH_MAPPINGS` 映射。

示例：
- 环境变量：`PATH_MAPPINGS=C:/Users/yourname/data/piping_inputs=/data/local_input`
- 请求：`C:\Users\yourname\data\piping_inputs\a.jpg`
- 实际处理：`/data/local_input/a.jpg`

要求：
- `LOCAL_INPUT_DIR` 要挂载到容器（compose 已支持）
- `PATH_MAPPINGS` 前缀要与客户端传入路径一致

## 5. 响应输出（Server -> Client）

服务端按顺序返回：

### 5.1 ack

```json
{
  "type": "ack",
  "status": "accepted",
  "task_id": "task-001"
}
```

### 5.2 progress（多条）

```json
{
  "type": "progress",
  "status": "running",
  "task_id": "task-001",
  "progress": 35,
  "stage": "infer_image",
  "message": "processing image 2/5"
}
```

### 5.3 result（成功）

```json
{
  "type": "result",
  "status": "ok",
  "task_id": "task-001",
  "result": {
    "task_id": "task-001",
    "created_at": "2026-03-28T10:00:00.000000Z",
    "request_file_path": "C:\\Users\\yourname\\data\\piping_inputs\\demo.jpg",
    "resolved_input_path": "/data/local_input/demo.jpg",
    "config": "/workspace/mmyolo/...",
    "checkpoint": "/workspace/mmyolo/...",
    "device": "cuda:0",
    "score_thr": 0.3,
    "result": {}
  }
}
```

### 5.4 error（失败）

```json
{
  "type": "error",
  "status": "error",
  "task_id": "task-001",
  "message": "input file path not found: ...",
  "traceback": "..."
}
```

## 6. 结果体结构说明（`result.result`）

### 6.1 图片/图片目录任务

```json
{
  "kind": "image",
  "input": "/data/local_input",
  "total_images": 10,
  "output_jsonl": "/workspace/mmyolo/output/tcp_results/task-001_images.jsonl",
  "records": [
    {
      "image": "/data/local_input/a.jpg",
      "detections": [
        {
          "label_id": 0,
          "label_name": "xxx",
          "score": 0.923456,
          "bbox_xyxy": [100.12, 50.33, 200.66, 150.77]
        }
      ],
      "pred_level": [
        {
          "class_id": 0,
          "class_name": "xxx",
          "grade": 2
        }
      ]
    }
  ]
}
```

### 6.2 视频任务

```json
{
  "kind": "video",
  "input": "/data/local_input/demo.mp4",
  "total_frames": 3000,
  "processed_frames": 3000,
  "inferenced_frames": 1000,
  "infer_stride": 3,
  "output_jsonl": "/workspace/mmyolo/output/tcp_results/task-001_video.jsonl",
  "records": []
}
```

说明：
- 当 `return_records=false`，`records` 为空，逐帧结果在 `output_jsonl`。
- 当 `return_records=true`，`records` 返回逐帧检测结果，可能很大。

## 7. 调用示例（Python）

```python
import json
import socket

HOST = "127.0.0.1"
PORT = 9000

request = {
    "task_id": "task-001",
    "file_path": r"C:\Users\yourname\data\piping_inputs\demo.jpg",
    "return_records": False
}

with socket.create_connection((HOST, PORT), timeout=30) as sock:
    sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))

    buf = b""
    done = False
    while not done:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line:
                continue
            msg = json.loads(line.decode("utf-8"))
            print(msg)
            if msg.get("type") in {"result", "error"}:
                done = True
                break
```

## 8. 常见问题

- 报错“windows path cannot map to container path”  
  没配置或配置错 `PATH_MAPPINGS`，或目录未正确挂载。
- 报错“input file path not found”  
  容器内映射路径不存在，先检查挂载路径和文件是否真实存在。
- 视频结果回包过大  
  把 `return_records` 设为 `false`，从 `output_jsonl` 读取明细。
