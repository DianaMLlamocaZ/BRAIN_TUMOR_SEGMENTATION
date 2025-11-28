#Custom dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import math
import matplotlib.pyplot as plt
import albumentations as A

from read_imgs import read_img_json

class DatasetMI(torch.utils.data.Dataset):
    def __init__(self,path_dir,augment:bool,size=(256,256)):
        self.path_dir=path_dir
        self.size=size
        
        if augment:
            self.transform=A.Compose([
                A.Resize(self.size[0],self.size[1]),
                A.ToTensorV2()
            ])

        else:
            self.transform=A.Compose([
                A.Resize(self.size[0],self.size[1]),
                A.ToTensorV2()
            ])
        

    def __len__(self):
        return len(os.listdir(self.path_dir))-1


    def __getitem__(self,index):
        img_fn,annotation=read_img_json(self.path_dir,index) #name, bbox
        
        #Imagen
        img_pil=Image.open(f"{self.path_dir}/{img_fn}")
        img_numpy=(np.array(img_pil)/255).astype(np.float32)
        

        #Máscara: por default es 640,640
        mask_final=np.zeros(shape=(640,640)).astype(np.float32)


        #Mask delimitaciones
        x_sup,y_sup=int(math.floor(annotation[0])),int(math.floor(annotation[1]))

        x_dif,y_dif=int(math.ceil(annotation[2])),int(math.ceil(annotation[3]))
        x_inf,y_inf=int(x_sup+x_dif),int(y_sup+y_dif)


        #Final mask
        mask_final[y_sup:y_inf,x_sup:x_inf]=1


        #Preprocess de la data
        transformed=self.transform(image=img_numpy,mask=mask_final)
        final_img,final_mask=transformed["image"],transformed["mask"]


        return final_img[0,:,:].unsqueeze(0),final_mask.unsqueeze(0) #Solo 1 canal (porque los 3 canales son iguales)
