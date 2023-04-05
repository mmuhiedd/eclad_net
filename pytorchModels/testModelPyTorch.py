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
parser.add_argument('--nb_batch',help='nb batch to test. Usefull to test inference on only one batch. If 0, all batchs are proceed',type=int, default=0)

args = parser.parse_args()
isGhostNet = args.ghost
nb_batch = args.nb_batch

batchNorm = False

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

elif batchNorm:
    netType="ecladNet"
    model = Net()
    model.load_state_dict(torch.load('runs/model/pytorch/ecladNet.pt',map_location=device))

else:
    netType="EcladNet noBatch"
    model = NetNoBatch()
    model.load_state_dict(torch.load('runs/model/pytorch/ecladNet_noBatch.pt',map_location=device))


print("\n %s \n" % netType)
model.to(device)
model.eval()
acc = testModelPyTorch_acc(model, testloader,device, nb_batch)
metrics = testModelPyTorch_inferenceTime(model, testloader,device, nb_batch)

    


