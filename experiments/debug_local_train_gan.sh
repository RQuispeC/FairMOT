cd src
python train_gan.py mot_gan \
    --gpus -1 \
    --batch_size 2 \
    --arch dlasplit6gan_34 \
    --load_model '../models/crowdhuman_dla34.pth' \
    --load_on_generator \
    --num_epochs 1 \
    --data_root_dir '/mnt/c/Users/edquispe/OneDrive - Microsoft/Documents/mot/MOT15_1_sequence_copy' \
    --data_cfg '../src/lib/cfg/mot15_debug.json' \
    --save_dir 'exp/tmp_gan'

cd ..