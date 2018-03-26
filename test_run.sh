
MODEL_DIR=test_loop
BLOCK=van
DEPTH=1
FILE_DIR=TFRecords_224x224
TRAIN_STEPS=500
EVAL_STEPS=500
STOCH_RUNS=0
VERBOSITY=INFO # Slows things down. Remove for large scale training.



python run_deep_models.py \
        --model-dir $MODEL_DIR \
        --block $BLOCK \
        --depth $DEPTH \
        --resolution 224 224 \
        --file-dir $FILE_DIR \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --stoch-passes $STOCH_RUNS \
        --verbosity $VERBOSITY
