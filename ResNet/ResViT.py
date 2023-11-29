import torch.nn as nn
from Res18 import ResComp
from ViT_Component import ViT


class ResViT(nn.Module):
    def __init__(self, num_class = 4):
        super(ResViT, self).__init__()
        self.ResComp = ResComp()
        # # ResViT-Base
        # self.ViTComp = ViT(image_size = 27, 
        #                    patch_size = 3, 
        #                    num_classes = num_class, 
        #                    channels = 256, 
        #                    dim = 1024, 
        #                    depth = 1, 
        #                    heads = 4,
        #                    dim_head=64,
        #                    mlp_dim = 2048, 
        #                    dropout = 0.01, 
        #                    emb_dropout = 0.01, 
        #                    pool = "mean")
        # # ResViT-Tiny
        self.ViTComp = ViT(image_size = 27, 
                           patch_size = 3, 
                           num_classes = num_class, 
                           channels = 128, 
                           dim = 512, 
                           depth = 1, 
                           heads = 8,
                           dim_head = 32,
                           mlp_dim = 1024, 
                           dropout = 0.05, 
                           emb_dropout = 0.05, 
                           pool = "mean")

    def forward(self, x):
        x = self.ResComp(x)
        x = self.ViTComp(x)
        return x
        
        
