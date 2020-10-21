LABELED_DATA=../data/semeval/semeval_labeled.json
UNLABELED_DATA=../data/semeval/semeval_unlabeled.json
EXP_DATA=../data/semeval/semeval_exp.json
VALID_DATA=../data/semeval/semeval_valid_data.json
TEST_DATA=../data/semeval/semeval_test_data.json
OUTPUT_DIR=./checkpoints/s2s_ft_semeval_checkpoints_update2/
CACHE_DIR=../cache/s2s_ft_package_semeval_cache
CONFIG_FILE=./s2s_ft/config.json
LABEL_TO_ID=../data/semeval/label_to_id.json

export CUDA_VISIBLE_DEVICES=5
python main.py \
  --labeled_data ${LABELED_DATA} --unlabeled_data ${UNLABELED_DATA} --exp_data ${EXP_DATA} \
  --valid_data ${VALID_DATA} --test_data ${TEST_DATA} --model_update_steps 2\
  --num_labels 19 --output_dir ${OUTPUT_DIR} --train_from_scratch --label_to_id ${LABEL_TO_ID} \
  --do_lower_case --max_source_seq_length 128 --max_target_seq_length 128 \
  --num_classifier_epochs_per_iters 3 --num_generator_epochs_per_iters 5 \
  --per_gpu_train_batch_size 2 --gradient_accumulation_steps 16 \
  --num_iters 8 --num_selftrain_iters 20 --learning_rate 1e-5 --num_warmup_steps 500 --cache_dir ${CACHE_DIR}