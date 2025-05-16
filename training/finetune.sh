python3 ../aardvark/generate_initial_condition_single.py \
    --assimilation_model_path ENCODER_PATH

python3 ../aardvark/finetune.py \
    --assimilation_model_path ENCODER_PATH \
    --forecast_model_path FORECAST_PATH \
    --output_dir FINETUNE_PATH \
    --lr 5e-5 \
    --finetune_epochs 1