import torch
import argparse
from classes import *
from definitions import *
from torch.utils.data import DataLoader




##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule',type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule',type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)


args = parser.parse_args()
isGhostNet = args.ghost

# set device used :
## Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retrieve input 
img_list_test, class_list_test_t = read_images(train=False)
class_list_test_t = onehotencoding_class(class_list_test_t)


transformed_dataset_test = Logo_Dataset(img_list_test, class_list_test_t,
                                         transform = transforms.Compose([Rescale(32), ToTensor()]))
testloader = DataLoader(transformed_dataset_test, batch_size=32,
                        shuffle=True, pin_memory=True)

if isGhostNet:
    netType = "GhostNet"
    model = GhostNet(args.ratio1,args.ratio2)
    model.load_state_dict(torch.load('runs/model/pytorch/ghostNet_{}_{}.pt'.format(args.ratio1,args.ratio2),map_location=device))

else:
    netType="ecladNet"
    model = Net()
    model.load_state_dict(torch.load('runs/model/pytorch/ecladNet.pt',map_location=device))


model.to(device)
model.eval()
acc,time_inf = testModelPyTorch_InputToDevice_Once(model, testloader,device)

    


