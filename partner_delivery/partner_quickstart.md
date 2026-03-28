# MMYOLO TCP 服务（对接方快速使用说明）

这份文档给“只需要会用，不关心内部实现”的同学。

## 1. 你将拿到什么

请提供方（你）发给对接方这些内容：

1. 镜像地址：`registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0`
2. 启动文件：`deploy/docker-compose.tcp-infer.runtime.yaml`
3. `.env` 模板：`deploy/.env.runtime.template`
4. 测试脚本：`deploy/test_tcp_infer_client.py`
5. 权重文件（放到对接方机器的 `weights` 目录）
6. 一条可用的测试请求样例（文件路径示例）

## 2. 使用前准备

对接方机器需要：

1. 已安装 Docker Desktop（Windows）或 Docker Engine（Linux）
2. 磁盘里准备 3 个目录
   - `D:\piping\weights`（模型权重）
   - `D:\piping\input`（待分析文件）
   - `D:\piping\output`（输出结果）

## 3. 第一次启动（最简步骤）

### 步骤 1：准备配置文件

把 `deploy/.env.runtime.template` 复制为 `.env`，并按实际路径修改。

### 步骤 2：拉取镜像

公开镜像一般可以直接拉，不需要登录：

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0
```

如果报无权限，再执行登录：

```bash
docker login registry.cn-hangzhou.aliyuncs.com
docker pull registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0
```

### 步骤 3：启动服务

进入 `docker-compose.tcp-infer.runtime.yaml` 所在目录执行：

```bash
docker compose -f docker-compose.tcp-infer.runtime.yaml up -d
docker logs -f mmyolo-tcp-infer
```

看到 `server listening on 0.0.0.0:9000` 代表启动成功。

## 4. `.env` 怎么改

示例（Windows）：

```env
WEIGHTS_DIR=D:/piping/weights
LOCAL_INPUT_DIR=D:/piping/input
OUTPUT_DIR_HOST=D:/piping/output
PATH_MAPPINGS=D:/piping/input=/data/local_input
```

说明：

1. `WEIGHTS_DIR`：本机权重目录
2. `LOCAL_INPUT_DIR`：本机输入目录
3. `OUTPUT_DIR_HOST`：本机输出目录
4. `PATH_MAPPINGS`：把“请求里的 Windows 路径前缀”映射到容器路径

例如请求里传：
`D:\piping\input\demo.jpg`

服务会自动转换为容器内：
`/data/local_input/demo.jpg`

## 5. 如何验证可用

将测试图片放到 `D:\piping\input` 后执行：

```bash
python test_tcp_infer_client.py --host 127.0.0.1 --port 9000 --file-path "D:\piping\input\demo.jpg"
```

看到 `[OK] task finished` 表示链路正常。

## 6. 常用运维命令

启动：

```bash
docker compose -f docker-compose.tcp-infer.runtime.yaml up -d
```

查看日志：

```bash
docker logs -f mmyolo-tcp-infer
```

停止：

```bash
docker compose -f docker-compose.tcp-infer.runtime.yaml down
```

重启：

```bash
docker restart mmyolo-tcp-infer
```

## 7. 常见问题（简版）

1. 需要 `docker login` 吗？  
公开镜像通常不需要。先直接 `docker pull`，失败再登录。

2. 报“找不到文件”  
优先检查：`.env` 的 `LOCAL_INPUT_DIR` 和 `PATH_MAPPINGS` 是否与请求路径前缀一致。

3. 没有 GPU 怎么办  
把 compose 里的 `MODEL_DEVICE` 改成 `cpu`，并删除 `gpus: all`。
