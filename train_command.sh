# TODO: change this to your need

python finetune.py --arch vit  \
                   --project "Vit on TinyImageNet" \
                   --name "0005 Train TinyImageNet on local PC" \
                   --dataset Tiny-Imagenet \
                   --datapath /home/long/data/tiny-imagenet-200 \
                   --batch_size 128 \
                   --epochs 100 \
                   --gpu 0