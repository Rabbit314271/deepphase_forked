import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import random
from Encoder import Encoder
import Library.Utility as utility

class PAE(nn.Module):
    def __init__(self, input_channels, embedding_channels, time_range, key_range, window, full_range,enc_path=''):
        super(PAE, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.key_range = key_range
        self.full_range=full_range

        self.window = window
        self.time_scale = key_range/time_range

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * (time_range * self.time_scale) / self.window, requires_grad=False) #Remove DC frequency

        

        intermediate_channels = int(input_channels/3)
        
        #self.encs = torch.nn.ModuleList()
        self.enc = utility.ToDevice(Encoder(input_channels, embedding_channels, time_range, key_range, window))
        if enc_path!='':
            self.enc = self.load_params(self.enc,enc_path)
            #self.enc.requires_grad_=False
        
        #for i in range(self.full_range-self.time_range+1):
        #    self.encs.append(self.enc)

        
        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_deconv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    def load_params(self,model,path):
        pretrained_dict = torch.load(path).state_dict()
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        #   3. load the new state dict
        model.load_state_dict(model_dict)

        return model

    def merge(self,clips,motion_dim,clip_dim):
        n_curves=motion_dim-clip_dim+1

        signal=torch.zeros([1,8,motion_dim]).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for i in np.arange(n_curves):
            signal[0,:,i:i+clip_dim]+=clips[i,:,:]
            
        if motion_dim>2*clip_dim:
            for i in np.arange(motion_dim):
                if (i<clip_dim):
                    signal[:,:,i]/=i+1
                elif i>motion_dim-clip_dim:
                    signal[:,:,i]/=motion_dim-i
                else:
                    signal[:,:,i]/=clip_dim
        else:
            for i in np.arange(motion_dim):
                if (i<motion_dim-clip_dim):
                    signal[:,:,i]/=i+1
                elif i>=clip_dim:
                    signal[:,:,i]/=motion_dim-i
        
                else:
                    signal[:,:,i]/=motion_dim-clip_dim+1
                    
        return signal



    def slices(self,x,motion_dim,clip_dim):
        n_curves=motion_dim-clip_dim+1

        p=x[:,:,0:clip_dim]
        #print(n_curves)
        for i in np.arange(n_curves-1):
            #print('p shape:',p.shape,'one x shape:',x[:,:,i+1:i+clip_dim+1].shape)
            p=torch.concat((p,x[:,:,i+1:i+clip_dim+1]),0)
            
        return p

    def padding(self,a,n_padding):
    
        
        #print('n_padding',n_padding)
    
        a=torch.concat([torch.flip(torch.flip(a,dims=[2])[:,:,:n_padding],dims=[2]),a,a[:,:,:n_padding]],2)
        return a

    def forward(self, x):
        batch_size=x.shape[0]
        padding_num=int((self.time_range-1)/2)
        x=self.padding(x,padding_num)
        #print('should be [b,72,100]',x.shape)
        y=[]
        #x:[b,72,100]
        full_range=self.time_range-1+self.full_range#should be 100
        n_curves=full_range-self.time_range+1#should be 60
        
        #print()

        for i in np.arange(batch_size):
            y.append(self.slices(x[i,:,:].unsqueeze(0),full_range,self.time_range))
        y=torch.concat(y,0)
        #[60*b,72,41]
        #Signal Embedding
        #print('\nafter slice should be:[60*b,72,41]',y.shape)
        
        p,f,a,b,latent=self.enc(y)
        
        params=[p[::n_curves,:,:],f[::n_curves,:,:],a[::n_curves,:,:],b[::n_curves,:,:]]
        
        #param:[4,20*b,8,1]
        #latent:[20*b,8,41]

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        #[20*b,8,41]
        #print('\nafter signalize should be:[60*b,72,41]',y.shape)
        signal=[]
        for i in np.arange(0,batch_size):
            signal.append(self.merge(y[i*n_curves:i*n_curves+n_curves],full_range,self.time_range))
            #print('\n one signal',signal[-1].shape)
             #Save signal for returning
        signal=torch.concat(signal,0)
        #print('\nafter average should be:[b,72,100]',signal.shape)
        y=signal[:,:,padding_num:-padding_num]
        #(b,8,60)
        #print('y shape should be [b,8,60]',y.shape)
        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.bn_deconv1(y)
        y = torch.tanh(y)
        #(b,24,60)
        y = self.deconv2(y)
        #print('\nafter deconv should be:[b,24,60]',signal.shape)
        #(b,72,60)
        #y = y.reshape(y.shape[0], self.input_channels*self.time_range)
        yplot=y[:,:,:self.time_range]
        #print(y.shape)

        return y, latent[::n_curves,:,:], signal[:,:,:self.time_range], params,yplot