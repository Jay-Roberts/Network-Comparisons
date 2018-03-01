
MODEL_DIR=test
BLOCK=van
DEPTH=3
FILE_DIR=TFRecords_28x28
TRAIN_STEPS=2000
EVAL_STEPS=1000




python3 run_deep_models.py \
        --model-dir $MODEL_DIR \
        --block $BLOCK \
        --depth $DEPTH \
        --resolution 28 28 \
        --file-dir $FILE_DIR \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS
