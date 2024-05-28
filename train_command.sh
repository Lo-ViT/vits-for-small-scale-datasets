# TODO: change this to your need

python finetune.py --arch vit-ats  \
                   --output_dir /graphics/scratch2/students/nguyenlo \
                   --project "Vit on TinyImageNet" \
                   --name "0008 ViT ATS" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/students/nguyenlo/tiny-imagenet-200 \
                   --batch_size 128 \
                   --epochs 100 \
                   --gpu 0