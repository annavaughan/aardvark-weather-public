# Command for training with wind speed variable (ws)
python3 ../aardvark/e2e_train.py \
    --output_dir E2E/WS \
    --loss downscaling_rmse \
    --region global \
    --lead_time 1 \
    --sf_model_path DECODER_PATH \
    --se_model_path ENCODER_PATH \
    --forecast_model_path PROCESSOR_PATH \
    --var ws

# Command for training with temperature variable (tas)
python3 ../aardvark/e2e_train.py \
    --output_dir E2E/TAS \
    --loss downscaling_rmse \
    --region global \
    --lead_time 1 \
    --sf_model_path DECODER_PATH \
    --se_model_path ENCODER_PATH \
    --forecast_model_path PROCESSOR_PATH \
    --var tas