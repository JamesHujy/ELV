LABELED_DATA=../data/restaurants/restaurants_exp.json
UNLABELED_DATA=../data/restaurants/restaurants_train_data.json
VALID_DATA=../data/restaurants/restaurants_valid_data.json
TEST_DATA=../data/restaurants/restaurants_test_data.json
OUTPUT_DIR=./checkpoints/s2s_ft_restaurants_checkpoints/
CACHE_DIR=../cache/s2s_ft_package_restaurants_cache
CONFIG_FILE=./s2s_ft/config.json

export CUDA_VISIBLE_DEVICES=2
python main.py \
  --labeled_data ${LABELED_DATA} --unlabeled_data ${UNLABELED_DATA} \
  --valid_data ${VALID_DATA} --test_data ${TEST_DATA} \
  --num_labels 3 --output_dir ${OUTPUT_DIR} \
  --do_lower_case --max_source_seq_length 128 --max_target_seq_length 128 \
  --num_classifier_epochs_per_iters 3 --num_generator_epochs_per_iters 5 \
  --per_gpu_train_batch_size 8 --gradient_accumulation_steps 4 \
  --num_iters 20 --learning_rate 1e-5 --num_warmup_steps 500 --cache_dir ${CACHE_DIR}