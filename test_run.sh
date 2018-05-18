
MODEL_DIR=d_model
BLOCK=Wf_EM
DEPTH=1
FILE_DIR=cifar-10-data
TRAIN_STEPS=10
EVAL_STEPS=5
STOCH_RUNS=2
VERBOSITY=INFO # Slows things down. Remove for large scale training.



python3 run_deep_models.py \
        --test cifar \
        --model-dir $MODEL_DIR \
        --block $BLOCK \
        --depth $DEPTH \
        --resolution 32 32 \
        --file-dir $FILE_DIR \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --stoch-passes $STOCH_RUNS \
        --verbosity $VERBOSITY
