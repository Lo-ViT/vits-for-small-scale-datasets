# TODO: change this to your need

python finetune.py --arch vit  \
                   --project "Example project" \
                   --name "Example experiment" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/datasets/tiny-imagenet-200/tiny-imagenet-200 \
                   --batch_size 512 \
                   --epochs 100 \
                   --gpu 0