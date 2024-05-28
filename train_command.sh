# TODO: change this to your need

python finetune.py --arch vit-ats \
                   --pretrained_weights vit_timnet_patch8_input64.pth \
                   --output_dir /graphics/scratch2/students/nguyenlo \
                   --project "Vit on TinyImageNet" \
                   --name "0011 ViT ATS with pretraining" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/students/nguyenlo/tiny-imagenet-200 \
                   --batch_size 128 \
                   --epochs 100 \
                   --gpu 6