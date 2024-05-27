# TODO: change this to your need
export CUDA_VISIBLE_DEVICES=4,5,6,7

python finetune.py --arch vit  \
                   --output_dir /graphics/scratch2/students/nguyenlo \
                   --project "Vit on TinyImageNet" \
                   --name "0007 Multiple GPUs training" \
                   --dataset Tiny-Imagenet \
                   --datapath /graphics/scratch2/students/nguyenlo/tiny-imagenet-200 \
                   --batch_size 512 \
                   --epochs 100 \
                   --gpu 0