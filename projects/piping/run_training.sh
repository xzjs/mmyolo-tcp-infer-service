WORK_DIR_ROOT=$1
CONFIG=$2

PYTHON="/home/zhuangwj/miniconda3/envs/openmmlab/bin/python"
cd /home/zhuangwj/mmyolo

${PYTHON} tools/train.py --config ${CONFIG} --work-dir=${WORK_DIR_ROOT}/stage1 --cfg-options model.backbone.frozen_stages=4 --cfg-options model.neck.freeze_all=True --amp

checkpoint=$(ls ${WORK_DIR_ROOT}/stage1/best*)
${PYTHON} tools/train.py --config ${CONFIG} --work-dir=${WORK_DIR_ROOT}/stage2 --cfg-options model.backbone.frozen_stages=4 --cfg-options optim_wrapper.optimizer.lr=0.001 --cfg-options load_from=${checkpoint} --amp

checkpoint=$(ls ${WORK_DIR_ROOT}/stage2/best*)
${PYTHON} tools/train.py --config ${CONFIG} --work-dir=${WORK_DIR_ROOT}/stage3 --cfg-options model.backbone.frozen_stages=3 --cfg-options optim_wrapper.optimizer.lr=0.0001 --cfg-options load_from=${checkpoint} --amp

checkpoint=$(ls ${WORK_DIR_ROOT}/stage3/best*)
${PYTHON} tools/train.py --config ${CONFIG} --work-dir=${WORK_DIR_ROOT}/stage4 --cfg-options model.backbone.frozen_stages=2 --cfg-options optim_wrapper.optimizer.lr=0.00001 --cfg-options load_from=${checkpoint} --amp

checkpoint=$(ls ${WORK_DIR_ROOT}/stage4/best*)
${PYTHON} tools/train.py --config ${CONFIG} --work-dir=${WORK_DIR_ROOT}/stage5 --cfg-options model.backbone.frozen_stages=1 --cfg-options optim_wrapper.optimizer.lr=0.00001 --cfg-options load_from=${checkpoint} --amp

checkpoint=$(ls ${WORK_DIR_ROOT}/stage5/best*)
${PYTHON} tools/train.py --config ${CONFIG} --work-dir=${WORK_DIR_ROOT}/stage6 --cfg-options optim_wrapper.optimizer.lr=0.000001 --cfg-options load_from=${checkpoint} --amp
