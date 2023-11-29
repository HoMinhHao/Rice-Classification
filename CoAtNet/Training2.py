from Settings2 import *
from adabelief_pytorch import AdaBelief
from CoAtNet_Mish import *
from ranger_adabelief import RangerAdaBelief
# from torch.nn.parallel import DataParallel

Arch = ['TinyResViTV3','resnet18','resnext50_32x4d','densenet121','deit_tiny_patch16_224','gmlp_ti16_224','mixer_s16_224']
Dataset = ['CornPV', 'CornBangla']
Device = torch.device('cuda:0')

if __name__ == '__main__':
    # for ID in range(0,1):
    #     f = open(f"./TrainingResult/{Arch[ID]}.txt","w")
    #     print('Start building Model...')
    #     if (Arch[ID] == 'TinyResViTV3'):
    #         Model = ResViT(num_class = 8)
    #     else:
    #         Model = timm.create_model(Arch[ID], num_classes=8)
        Model=coatnet_0(num_classes=8)
        Model.to(Device)
        print(Model)
        print('Build Model successfully!')
        train_loader, val_loader = prepare_dataloader()

        # # For Transformer, MLP
        # https://github.com/juntang-zhuang/Adabelief-Optimizer#table-of-hyper-parameters
        # optimizer = AdaBelief(Model.parameters(), lr=1*1e-5, eps=1e-16, betas=(0.9,0.999), weight_decay=1e-4, 
        #                       weight_decouple = True, rectify = True, fixed_decay = False, amsgrad = False)
        # optimizer = torch.optim.Adam(Model.parameters(),lr=0.0001,eps=1e-8,amsgrad=True)
        optimizer = RangerAdaBelief(Model.parameters(), lr=0.001, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=5*1e-7)

        # # For ResNext, DenseNet
        # optimizer = AdaBelief(Model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decay=5e-4, 
        #                       weight_decouple = False, rectify = False, fixed_decay = False, amsgrad = False)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-5)
        
        
        loss_tr = nn.CrossEntropyLoss().to(Device)
        loss_vl = nn.CrossEntropyLoss().to(Device)
        ValidAccuracy = []
        ValidLoss = []

        # early_stopping = EarlyStopping("ConViT", patience=20)
        for epoch in range(CFG['epochs']):
            print('=================================================')
            print(f'\n[ TRAINING EPOCH {epoch} ]')
            history=TrainAndEval(epoch, Model, loss_tr, optimizer, train_loader, val_loader, Device, scheduler=scheduler,schd_batch_update=True)
            
        torch.save(history, "Models/CoatNet_result_py.txt")
        f.close()