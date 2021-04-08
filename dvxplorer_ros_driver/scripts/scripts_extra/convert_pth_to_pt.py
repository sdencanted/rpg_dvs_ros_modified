from utils.loading_utils import load_model, get_device
import torch
from model.model import *

path = './pretrained/firenet_1000.pth.tar'
raw_model = torch.load(path)
arch = raw_model['arch']
model_type = raw_model['config']['model']
model = eval(arch)(model_type)

#model_dict = {'epoch':raw_model['epoch'],'model':model,'optimizer':raw_model['optimizer']}

torch.save(model.state_dict(),'firenet.pt')
