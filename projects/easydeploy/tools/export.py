# 导入必用依赖
import tensorrt as trt
# 创建logger：日志记录器
logger = trt.Logger(trt.Logger.WARNING)

onnx_path = 'work_dir/best_coco_bbox_mAP_epoch_50.onnx'
engine_path = 'work_dir/best_coco_bbox_mAP_epoch_50.engine'

# 创建构建器builder
builder = trt.Builder(logger)
# 预创建网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# 加载onnx解析器
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(onnx_path)
for idx in range(parser.num_errors):
  print(parser.get_error(idx))
if not success:
  pass  # Error handling code here
# builder配置
config = builder.create_builder_config()
# 分配显存作为工作区间，一般建议为显存一半的大小
# config.max_workspace_size = 1 << 30  # 1 Mi
#print(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE))
#config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
#print(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE))
serialized_engine = builder.build_serialized_network(network, config)
# 序列化生成engine文件
with open(engine_path, "wb") as f:
   f.write(serialized_engine)
   print("generate file success!")
