# TODO: change this to your need

python test.py --arch vit  \
                   --model "/graphics/scratch2/students/nguyenlo/CIFAR10/0012 No Layers ATS - drop_tokens=false/save_finetuned/vit-Base0012 No Layers ATS - drop_tokens=false--CIFAR10-LR[0.001]-Seed0/best.pth" \
                   --dataset CIFAR10 \
                   --batch_size 256 \
                   --output_dir /graphics/scratch2/students/nguyenlo/