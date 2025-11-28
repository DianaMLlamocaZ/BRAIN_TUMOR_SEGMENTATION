import math
import torch

#Función para que los shapes del encoder y decoder, al concatenar, sean los mismos
def crop(fm_enc,fm_dec):
  
  #Calculo la diferencia de h y w
  dh=math.ceil((fm_enc.size(2)-fm_dec.size(2))//2)
  dw=math.ceil((fm_enc.size(3)-fm_dec.size(3))//2)
  
  #Recorto la img (una distancia central)
  if (fm_enc.size(2)-fm_dec.size(2))%2==0:
    fm_enc_cropped=fm_enc[:,:,dh:fm_enc.size(2)-dh,dw:fm_enc.size(3)-dw]
  else:
    fm_enc_cropped=fm_enc[:,:,dh:fm_enc.size(2)-dh-1,dw:fm_enc.size(3)-dw-1]
 
  return fm_enc_cropped


#Doble convolución en una capa de la UNet
class DobleConv(torch.nn.Module):
  def __init__(self,channels_in,channels_out):
    super().__init__()
    self.cnn1=torch.nn.Conv2d(in_channels=channels_in,out_channels=channels_out,kernel_size=3,padding="same")
    self.cnn2=torch.nn.Conv2d(in_channels=channels_out,out_channels=channels_out,kernel_size=3,padding="same")
    self.r=torch.nn.ReLU()

  def forward(self,data):
    x1=self.r(self.cnn1(data))

    x2=self.r(self.cnn2(x1))

    return x2
  

#DownSampling: MaxPooling
class DownSampling(torch.nn.Module):
  def __init__(self,channels_in,channels_out):
    super().__init__()
    self.dc=DobleConv(channels_in,channels_out)
    self.mp=torch.nn.MaxPool2d(kernel_size=2)

  def forward(self,data):
    x_dc=self.dc(data)
    x_mp=self.mp(x_dc)
    return x_dc,x_mp
  

#UpSampling
class UpSampling(torch.nn.Module):
  def __init__(self,channels_in):
    super().__init__()
    self.us=torch.nn.ConvTranspose2d(in_channels=channels_in,out_channels=int(channels_in/2),kernel_size=4,stride=2,padding=1)
    self.cnn=torch.nn.Conv2d(in_channels=int(channels_in/2),out_channels=int(channels_in/2),kernel_size=3,padding="same") #padding same
    self.r=torch.nn.ReLU()

  def forward(self,data,concat):

    #UpSampling + conv
    x1=self.us(data) 
    x2=self.cnn(x1)

    #Igualar shapes
    concat_final=crop(concat,x2)

    #Concatenación
    x_concat=torch.concat([x2,concat_final],dim=1)
    
    return x_concat
  

#UNet modelo final
class UNet(torch.nn.Module):
  def __init__(self,channels_in,channels_out):
    super().__init__()
    #DownSampling
    self.ds1=DownSampling(channels_in,channels_out)
    self.ds2=DownSampling(channels_out,channels_out*2)
    self.ds3=DownSampling(channels_out*2,channels_out*4)

    #BottleNeck
    self.bn=DobleConv(channels_in=channels_out*4,channels_out=channels_out*4)

    #UpSampling
    self.us1=UpSampling(channels_out*4)
    self.us2=UpSampling(channels_out*4+channels_out*2)
    self.us3=UpSampling(channels_out*4+channels_out*1)

    #Final layer
    self.l_f=torch.nn.Conv2d(in_channels=channels_out*4-int(channels_out*1/2),out_channels=1,kernel_size=3,padding="same")


  def forward(self,data):
    #DS
    fm1,mp1=self.ds1(data) #fm1: Concatenar con el UpSampling del decoder en esa capa
    fm2,mp2=self.ds2(mp1) #fm2 Concatenar
    fm3,mp3=self.ds3(mp2) #fm3: Concatenar 


    #BottleNeck
    x_f=self.bn(mp3)
    

    #UpSampling
    x1_us=self.us1(x_f,fm3)
    x2_us=self.us2(x1_us,fm2)
    x3_us=self.us3(x2_us,fm1)
   
    #Conv final para tener 1 canal de salida
    conv_final=self.l_f(x3_us)

    return conv_final

