from .vit import VisionTransformer
from .swin import SwinTransformer
from .cait import cait_models
from functools import partial
from torch import nn
import sys
sys.path.insert(0,'submodules/ATS')
from submodules.ATS.libs.models.transformers.vit import ViT as ViT_ats

def create_model(img_size, n_classes, args):
    # CIFAR
    # 32 x 32
    # patch size = 4
    # (32 / 4)^2 = 64 tokens
    if args.arch == "vit-ats":
        patch_size = 4 if img_size == 32 else 8   #4 if img_size = 32 else 8
        model = ViT_ats(
            img_size = img_size,
            patch_size = patch_size,
            num_classes = n_classes,
            embed_dim=192,
            depth = 9,
            num_heads = 12,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=True,
            drop_path_rate=args.sd,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ats_blocks=[1,2,3,4,5,6,7,8],
            num_tokens=[65] * 9,
            drop_tokens=True,
        )

    elif args.arch == "vit":
        patch_size = 4 if img_size == 32 else 8   #4 if img_size = 32 else 8
        model = ViT_ats(
            img_size = img_size,
            patch_size = patch_size,
            num_classes = n_classes,
            embed_dim=192,
            depth = 9,
            num_heads = 12,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=True,
            drop_path_rate=args.sd,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ats_blocks=[],
            num_tokens=[65] * 9,
            drop_tokens=True,
        )

    elif args.arch == 'cait':       
        patch_size = 4 if img_size == 32 else 8
        model = cait_models(
        img_size= img_size,patch_size=patch_size, embed_dim=192, depth=24, num_heads=4, mlp_ratio=args.vit_mlp_ratio,
        qkv_bias=True,num_classes=n_classes,drop_path_rate=args.sd,norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,depth_token_only=2)
    
        
    elif args.arch =='swin':
        
        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if img_size==32 else 4

        model = SwinTransformer(img_size=img_size,
        window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],num_classes=n_classes,
       	mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.sd)

    else:
        NotImplementedError("Model architecture not implemented . . .")

         
    return model