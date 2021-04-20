from utils.loading_utils import load_model, get_device
import torch
from model.model import *

path = './pretrained/firenet_1000.pth.tar'

model = load_model(path)

for x in model.parameters():
    x.requires_grad = False

device = get_device('True')
model = model.to(device)
model.half()
model.eval()



curr_state = torch.zeros((1,5,240,320),dtype=torch.float16)
curr_state = curr_state.to(device)

prev_state1 = torch.zeros((1,16,240,320),dtype=torch.float16)
prev_state1 = prev_state1.to(device)


prev_state2 = torch.zeros((1,16,240,320),dtype=torch.float16)
prev_state2 = prev_state2.to(device)

prev_state = [prev_state1,prev_state2]



ONNX_PATH = './pretrained/firenet_1000_half.onnx'

torch.onnx.export(model, (curr_state,prev_state), ONNX_PATH, input_names=['input','prev_state_1','prev_state_2'],output_names=['output','new_state_1','new_state_2'],
                export_params=True)
