
MODEL_DIR=test_loop
BLOCK=van
DEPTH=3
FILE_DIR=TFRecords/TFRecords_28x28
TRAIN_STEPS=2
EVAL_STEPS=2
STOCH_RUNS=10



python run_deep_models.py \
        --model-dir $MODEL_DIR \
        --block $BLOCK \
        --depth $DEPTH \
        --resolution 28 28 \
        --file-dir $FILE_DIR \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS
