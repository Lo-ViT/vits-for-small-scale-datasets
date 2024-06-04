# TODO: change this to your need

python finetune.py --arch vit-ats  \
                   --output_dir /graphics/scratch2/students/nguyenlo \
                   --project "CIFAR10" \
                   --name "0010 Improve ATS on CIFAR" \
                   --dataset CIFAR10 \
                   --batch_size 256 \
                   --epochs 100 \
                   --gpu 0 \
                   --nowandb