# TODO: change this to your need

python test.py --arch vit  \
                   --model ~"/data/Baseline ViT-ATS CIFAR10/0001/save_finetuned/vit-ats-Base0001--CIFAR10-LR[0.001]-Seed0/best.pth" \
                   --project "Baseline ViT-ATS CIFAR10" \
                   --name 0001 \
                   --dataset CIFAR10 \
                   --batch_size 256 \
                   --epochs 100 \
                   --output_dir ~/data/