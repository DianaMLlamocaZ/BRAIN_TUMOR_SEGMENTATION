from torch.utils.data import DataLoader
import torch

from custom_dataset import DatasetMI
from model1 import SkipC_CNN
from model2 import UNet
from metrics import dice_loss

import random, numpy as np


def train(path_train,path_valid,batch_size_train,batch_size_test,epocas:int,device):
    #set_seed(SEED)
    #Moodelo y datasets
    modelo=UNet(channels_in=1,channels_out=16) #SkipC_CNN(in_channels=1)
    modelo=modelo.to(device)
    train_ds,valid_ds=DatasetMI(path_dir=path_train,augment=True,size=(256,256)),DatasetMI(path_dir=path_valid,augment=True,size=(256,256))


    #Dataloaders
    train_dl=DataLoader(train_ds,batch_size=batch_size_train,shuffle=True)
    valid_dl=DataLoader(valid_ds,batch_size=batch_size_test,shuffle=False)

    
    #Loss:BCE_WL
    bce_wl=torch.nn.BCEWithLogitsLoss()

    
    #Optimizer
    optimizer=torch.optim.Adam(modelo.parameters(),lr=1e-4)


    #Losses
    loss_train=[]
    loss_val=[]

    
    #Training
    for epoca in range(epocas):
        train_loss_epoca,valid_loss_epoca=[],[]

        for img_t_batch,mask_t_batch in train_dl:
            img_t_batch,mask_t_batch=img_t_batch.to(device),mask_t_batch.to(device)

            mask_pred_batch=modelo(img_t_batch).to(device)

            error=dice_loss(mask_pred_batch,mask_t_batch) #bce_wl
            train_loss_epoca.append(error.item())

            optimizer.zero_grad()

            error.backward() #Calcular gradientes
            optimizer.step() #Actualizar parámetros


        #Validation loss
        with torch.no_grad():
            for img_test_batch,mask_test_batch in valid_dl:
                img_test_batch,mask_test_batch=img_test_batch.to(device),mask_test_batch.to(device)
                    
                mask_test_pred=modelo(img_test_batch).to(device)
                    
                error_valid=dice_loss(mask_test_pred,mask_test_batch) #bce_wl
                valid_loss_epoca.append(error_valid.item())

        print(f"Época {epoca}: Train loss: {np.mean(np.array(train_loss_epoca))}. Valid loss: {np.mean(np.array(valid_loss_epoca))}")       

        loss_train.append(np.mean(np.array(train_loss_epoca)))
        loss_val.append(np.mean(np.array(valid_loss_epoca)))
        
    torch.save(modelo.state_dict(),"./modelos/modelo.pth")
    print("¡Modelo guardado!")
    
    return modelo,loss_train,loss_val
