# Online Continual Learning on Class Incremental Blurry Task Configuration with Anytime Inference
Code Analysis

## 명령어 (run)
bash scripts/clib.sh

## clib.sh
python main.py --mode $MODE --dataset $DATASET --n_tasks $N_TASKS --m $M --n $N --rnd_seed $RND_SEED --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME --lr $LR --batchsize $BATCHSIZE --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP

## main.py
* contiguration/config.py에 있는 argument 불러오기
* 로깅 설정 불러오기
* criterion, transforms, mode, ... 설정
* 데이터셋 불러오기
*