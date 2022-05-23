cd src
python eval.py mot\
    --gpus -1 \
    --batch_size 2 \
    --arch dlasplit3_34 \
    --load_model '../../fairmot-results/pleasing-koala/MMPTracking_sample_rate_4_cameras_1_train_split3/model_pleasing-koala.pth' \
    --data_root_dir '/mnt/c/Users/edquispe/OneDrive - Microsoft/Documents/mot/MOT15_1_sequence_copy' \
    --data_cfg '../src/lib/cfg/mot15_debug.json' \
    --save_dir 'exp/tmp_eval' \
    --save_images true
cd ..