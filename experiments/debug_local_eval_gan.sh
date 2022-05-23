cd src
python eval.py mot\
    --gpus -1 \
    --batch_size 2 \
    --arch dla2gan_34 \
    --load_model '../models/mot20_fairmot.pth' \
    --data_root_dir '/mnt/c/Users/edquispe/OneDrive - Microsoft/Documents/mot/MOT15_1_sequence_copy' \
    --data_cfg '../src/lib/cfg/mot15_debug.json' \
    --save_dir 'exp/tmp_eval_gan'

cd ..