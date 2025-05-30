python3 ../aardvark/train_module.py \
    --output_dir DECODER/TAS \
    --master_port 22348 \
    --decoder base \
    --loss downscaling_rmse \
    --diff 0 \
    --in_channels 36 \
    --out_channels 1 \
    --int_channels 24 \
    --mode downscaling \
    --lr 5e-4 \
    --batch_size 64 \
    --start_ind 0 \
    --end_ind 24 \
    --epoch 20 \
    --weight_decay 1e-6 \
    --downscaling_context aardvark \
    --downscaling_train_start_date 2007-01-02 \
    --downscaling_train_end_date 2017-12-31 \
    --lead_time 4 \
    --var tas

python3 ../aardvark/train_module.py \
    --output_dir DECODER/WS \
    --master_port 22348 \
    --decoder base \
    --loss downscaling_rmse \
    --diff 0 \
    --in_channels 36 \
    --out_channels 1 \
    --int_channels 24 \
    --mode downscaling \
    --lr 5e-4 \
    --batch_size 64 \
    --start_ind 0 \
    --end_ind 24 \
    --epoch 20 \
    --weight_decay 1e-6 \
    --downscaling_context aardvark \
    --downscaling_train_start_date 2007-01-02 \
    --downscaling_train_end_date 2017-12-31 \
    --lead_time 4 \
    --var ws