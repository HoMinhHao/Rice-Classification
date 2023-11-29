from Conv_module import *
from ViT_module import ViT

class ConViT(nn.Module):
    def __init__(self, num_class = 8):
        super(ConViT, self).__init__()
        self.ConvModule = ConvNeuralNet()
        self.ViTModule = ViT(image_size = 105, 
                           patch_size = 3, 
                           num_classes = num_class, 
                           channels = 64, 
                           dim = 512, 
                           depth = 1, 
                           heads = 8,
                           dim_head = 32,
                           mlp_dim = 1024, 
                           dropout = 0.05, 
                           emb_dropout = 0.05, 
                           pool = "mean")

    def forward(self, x):
        x = self.ConvModule(x)
        x = self.ViTModule(x)
        return x