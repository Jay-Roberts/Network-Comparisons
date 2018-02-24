#!/ /bin/sh

# Get the scripts
cp ../support_fns.py ../model_fns.py ../train_eval_exp.py .

# Set save dir
SAVE_DIR=Test_Models


# Set experiment parameters
MODEL_FN=SRNN_10
NUM_EPOCH=2
TRAIN_BATCH=10
EVAL_STEP=10
TRAIN_STEP=10

python train_eval_exp.py --model-fn $MODEL_FN \
			--save-dir $SAVE_DIR \
			--num-epochs $NUM_EPOCH \
			--train-batch-size $TRAIN_BATCH \
			--train-steps $TRAIN_STEP \
			--eval-steps $EVAL_STEP

# Clean up
rm support_fns.py model_fns.py train_eval_exp.py
echo $MODEL_FN": experiment complete"



