# TCP 镜像部署（无源码）

## 0. 镜像信息（阿里云）

- 镜像地址：`registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer`
- 版本：`1.0`
- 完整镜像：`registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0`

## 1. 直接拉取并运行（推荐）

### 1.1 登录阿里云镜像仓库

```bash
docker login registry.cn-hangzhou.aliyuncs.com
```

### 1.2 拉取镜像

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0
```

### 1.3 使用 runtime compose 启动

准备文件：
- `deploy/docker-compose.tcp-infer.runtime.yaml`
- `.env`（同目录，见下文示例）

启动：

```bash
docker compose -f docker-compose.tcp-infer.runtime.yaml up -d
docker logs -f mmyolo-tcp-infer
```

---

## 2. 离线包方式（无外网时）

### 2.1 在打包机上构建镜像

在源码目录执行：

```bash
docker build -f docker/Dockerfile_tcp_infer -t mmyolo-tcp-infer:latest .
```

如果需要指定基础镜像（例如你提前准备了本地基础镜像），可用：

```bash
docker build -f docker/Dockerfile_tcp_infer \
  --build-arg BASE_IMAGE=pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel \
  -t mmyolo-tcp-infer:latest .
```

### 2.2 导出镜像

```bash
docker save -o mmyolo-tcp-infer-latest.tar mmyolo-tcp-infer:latest
```

把这两个文件拷贝到目标机：
- `mmyolo-tcp-infer-latest.tar`
- `deploy/docker-compose.tcp-infer.runtime.yaml`

### 2.3 在目标机导入镜像

```bash
docker load -i mmyolo-tcp-infer-latest.tar
```

## 3. 准备目标机目录

示例（Windows）：
- `D:\piping\weights` 放权重文件
- `D:\piping\input` 放待分析文件
- `D:\piping\output` 放输出结果

## 4. 配置 `.env`

与 `docker-compose.tcp-infer.runtime.yaml` 同目录新建 `.env`：

```env
WEIGHTS_DIR=D:/piping/weights
LOCAL_INPUT_DIR=D:/piping/input
OUTPUT_DIR_HOST=D:/piping/output
PATH_MAPPINGS=D:/piping/input=/data/local_input
```

## 5. 启动服务

```bash
docker compose -f docker-compose.tcp-infer.runtime.yaml up -d
docker logs -f mmyolo-tcp-infer
```

## 6. 调用时传路径规则

TCP 请求里传 Windows 路径（必须在 `PATH_MAPPINGS` 范围内）：

```json
{
  "task_id": "task-001",
  "file_path": "D:\\piping\\input\\demo.jpg"
}
```

服务会自动映射为容器路径：
- `D:\piping\input\demo.jpg` -> `/data/local_input/demo.jpg`

## 7. 常见问题

- 目标机无 GPU  
  把 compose 中 `MODEL_DEVICE` 改为 `cpu`，并删除 `gpus: all`。
- 找不到文件  
  检查 `LOCAL_INPUT_DIR`、`PATH_MAPPINGS` 和请求里的 `file_path` 前缀是否一致。
- 构建时报 `403 Forbidden`（mirror.aliyuncs.com）  
  这是 Docker 镜像加速器无权限导致，不是项目代码问题。处理方式：
  1) Windows Docker Desktop -> `Settings` -> `Docker Engine`，移除无效 `registry-mirrors` 后重启 Docker。  
  2) 先单独测试拉取基础镜像：  
     `docker pull docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel`  
  3) 如果当前机器无法联网拉取，在可联网机器执行：  
     `docker pull docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel`  
     `docker save -o pytorch-base.tar docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel`  
     然后把 `pytorch-base.tar` 拷到打包机：  
     `docker load -i pytorch-base.tar`  
     再执行构建命令。

## 8. 部署后测试脚本

脚本位置：
- `deploy/test_tcp_infer_client.py`

调用示例（目标机）：

```bash
python test_tcp_infer_client.py --host 127.0.0.1 --port 9000 --file-path "D:\piping\input\demo.jpg"
```

可选：保存完整响应到本地文件

```bash
python test_tcp_infer_client.py --host 127.0.0.1 --port 9000 --file-path "D:\piping\input\demo.jpg" --save-response result.json
```
