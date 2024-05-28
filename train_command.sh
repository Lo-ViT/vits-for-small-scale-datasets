# TODO: change this to your need

python finetune.py --arch vit  \
                   --project "Vit on TinyImageNet" \
                   --name "In Memory Image Folder" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/datasets/tiny-imagenet-200/tiny-imagenet-200 \
                   --batch_size 128 \
                   --epochs 100 \
                   --gpu 4