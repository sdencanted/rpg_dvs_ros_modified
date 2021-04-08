from utils.loading_utils import load_model, get_device
import torch
from model.model import *
from torch2trt import torch2trt

path = './pretrained/firenet_1000.pth.tar'

model = load_model(path)

for x in model.parameters():
    x.requires_grad = False

device = get_device('True')
model = model.to(device)
model.eval()


curr_state = torch.zeros((1,5,240,320))
curr_state = curr_state.to(device)

prev_state1 = torch.zeros((1,16,240,320))
prev_state1 = prev_state1.to(device)


prev_state2 = torch.zeros((1,16,240,320))
prev_state2 = prev_state2.to(device)

prev_state = [prev_state1,prev_state2]

state = [curr_state,prev_state1,prev_state2]
model_trt = torch2trt(model,state,input_names=['input','prev_state_1','prev_state_2'],output_names=['output','new_state_1','new_state_2'],use_onnx=True)
a,b,c=model_trt(curr_state,prev_state1,prev_state2)
print(c)
torch.save(model_trt.state_dict(),'firenet_trt.pth')



# model_dict = {'epoch':raw_model['epoch'],'model':model.eval(),'optimizer':raw_model['optimizer']}

# torch.save(model_dict,'firenet.pt')
