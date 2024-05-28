# TODO: change this to your need

python finetune.py --arch vit  \
                   --project "Vit on TinyImageNet" \
                   --name "0006 In Memory Image Folder" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/students/nguyenlo/tiny-imagenet-200 \
                   --batch_size 128 \
                   --epochs 100 \
                   --gpu 4