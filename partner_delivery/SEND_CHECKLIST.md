# 发送清单（给对接方）

本文件夹已包含：
1. partner_quickstart.md（使用说明）
2. docker-compose.tcp-infer.runtime.yaml（启动文件）
3. .env.runtime.template（环境变量模板）
4. test_tcp_infer_client.py（连通性测试脚本）

你还需要额外提供：
1. 模型权重文件（请让对接方放到 WEIGHTS_DIR 指向目录）
2. 镜像地址：registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer:1.0
3. 一条可用输入路径样例（例如 D:\piping\input\demo.jpg）

提示：
- 镜像是公开仓库，通常可直接 docker pull；若提示鉴权失败再 docker login。
- 对接方可将 .env.runtime.template 复制为 .env 并按本机路径修改。
