TEST=cifar
BLOCK=Wf_EM
DEPTH=1
FILE_DIR=cifar-10-data
TRAIN_STEPS=1000
EVAL_STEPS=500
STOCH_RUNS=2
mkdir $TEST
cd $TEST
mkdir $BLOCK-$DEPTH-$FILE_DIR-$STOCH_RUNS-$VERBOSITY-model
cd ..
MODEL_DIR=$TEST/$BLOCK-$DEPTH-$FILE_DIR-$STOCH_RUNS-$VERBOSITY-model
VERBOSITY=INFO # Slows things down. Remove for large scale training.


echo $TEST $BLOCK $DEPTH $FILE_DIR $STOCH_RUNS $VERBOSITY > ./$MODEL_DIR/experiment_spec.txt



python3 run_deep_models.py \
        --test $TEST \
        --model-dir $MODEL_DIR \
        --block $BLOCK \
        --depth $DEPTH \
        --resolution 32 32 \
        --file-dir $FILE_DIR \
        --train-steps $TRAIN_STEPS \
        --eval-steps $EVAL_STEPS \
        --stoch-passes $STOCH_RUNS \
        --verbosity $VERBOSITY
