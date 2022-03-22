cd src
python train.py mot \
    --exp_id crowdhuman_dla34 \
    --gpus -1 \
    --batch_size 1 \
    --arch dla2_34 \
    --load_model '../models/crowdhuman_dla34.pth' \
    --num_epochs 1 \
    --data_root_dir '/mnt/c/Users/edquispe/OneDrive - Microsoft/Documents/mot/MOT15_1_sequence_copy' \
    --data_cfg '../src/lib/cfg/mot15_debug.json'
cd ..