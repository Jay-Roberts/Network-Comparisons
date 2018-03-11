
MODEL_DIR=test_loop
BLOCK=Sf_EM
DEPTH=2
FILE_DIR=TFRecords/TFRecords_28x28
TRAIN_STEPS=2
EVAL_STEPS=5
STOCH_RUNS=3
VERBOSITY=INFO # Slows things down. Remove for large scale training.



python run_deep_models.py \
        --model-dir $MODEL_DIR \
        --block $BLOCK \
        --depth $DEPTH \
        --resolution 28 28 \
        --file-dir $FILE_DIR \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --stoch-passes $STOCH_RUNS \
        --verbosity $VERBOSITY
