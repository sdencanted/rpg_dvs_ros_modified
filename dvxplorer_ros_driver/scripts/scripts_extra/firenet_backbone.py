import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init



class FireNet(nn.Module):

    def __init__(self, num_input_channels=5, num_output_channels=1, skip_type='no_skip',
                 recurrent_block_type='convgru', base_num_channels=16,
                 num_residual_blocks=2, norm=None, kernel_size=3,
                 recurrent_blocks={'resblock': [0]}):
        super(FireNet, self).__init__()

        self.kernel_size = kernel_size # 3
        self.recurrent_blocks = recurrent_blocks # {'resblock':[0]}


        self.head = ConvLayer(num_input_channels,base_num_channels,kernel_size=kernel_size)
        self.neck1 = ConvGRU(base_num_channels,base_num_channels,kernel_size=kernel_size)
        self.neck2 = Resblock(base_num_channels,base_num_channels,kernel_size=kernel_size)
        self.neck3 = ConvGRU(base_num_channels,base_num_channels,kernel_size=kernel_size)
        self.neck4 = Resblock(base_num_channels,base_num_channels,kernel_size=kernel_size)
        self.pred  = ConvLayer(base_num_channels,num_output_channels,kernel_size=1)
    
    def forward(self,x,prev_states):
        states =[]
        curr = 0
        
        x = self.head(x)
        state = self.neck1(x,prev_states[curr])
        states.append(state)
        curr += 1

        x = state

        x = self.neck2(x)
        state = self.neck3(x,prev_states[curr])
        states.append(state)
        curr += 1

        x  = state

        x = self.neck4(x)

        img = self.pred(x)

        return img,states

class ConvLayer(nn.Module):
    def __init__(self, input_channel,output_channel,kernel_size,stride=1):
        super(ConvLayer,self).__init__()

        padding = kernel_size//2

        self.conv2d = nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding=padding)

        self.activation = nn.ReLU(inplace=True)

    
    def forward(self,x):
        out = self.conv2d(x)

        out = self.activation(out)

        return out
    

class ConvGRU(nn.Module):
    def __init__(self,input_channel,hidden_size,kernel_size):
        super(ConvGRU,self).__init__()

        padding = kernel_size//2 
        self.input_size = input_channel
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(self.input_size+self.hidden_size,hidden_size,kernel_size,padding=padding)
        self.update_gate = nn.Conv2d(self.input_size+self.hidden_size,hidden_size,kernel_size,padding=padding)
        self.out_gate = nn.Conv2d(self.input_size+self.hidden_size,hidden_size,kernel_size,padding=padding)


        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    
    def forward(self,x,prev_state):

        if prev_state is None:
            size = x.size()
            prev_state = torch.zeros(size).to(x.device)
        
        input_stacked = torch.cat([x,prev_state],dim=1)
        update = torch.sigmoid(self.update_gate(input_stacked))
        reset = torch.sigmoid(self.reset_gate(input_stacked))
        reset = prev_state*reset
        reset = torch.cat([x,reset],dim=1)
        out = torch.tanh(self.out_gate(reset))
        out = update*out
        new_state = (1-update)*prev_state + out

        return new_state

class Resblock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,stride=1):
        super(Resblock,self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel,input_channel,kernel_size=kernel_size,stride=stride,padding=1)
        self.conv2 = nn.Conv2d(input_channel,input_channel,kernel_size=kernel_size,stride=stride,padding=1)
        
        

    
    def forward(self,x):

        out = f.relu(self.conv1(x),inplace=True)
        out = self.conv2(out)
        out += x
        out = f.relu(out,inplace=True)
        return out
        
