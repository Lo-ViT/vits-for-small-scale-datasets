python finetune.py --arch vit  \
                   --project "Vit on TinyImageNet" \
                   --name "0003 Larger Batch Size" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/datasets/tiny-imagenet-200/tiny-imagenet-200 \
                   --batch_size 256 \
                   --epochs 100 \
                   --gpu 0