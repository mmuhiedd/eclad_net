import torch
import argparse
from classes import *
from definitions import *
from torch.utils.data import DataLoader
import torch.profiler




##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule',type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule',type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)
parser.add_argument('--batch_size',help='Batch Size used. Default = 32', type=int, default=32)

args = parser.parse_args()
isGhostNet = args.ghost
batch_size = args.batch_size
# set device used :
## Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retrieve input 
img_list_test, class_list_test_t = read_images(train=False)
class_list_test_t = onehotencoding_class(class_list_test_t)


transformed_dataset_test = Logo_Dataset(img_list_test, class_list_test_t  , transform = transforms.Compose([Rescale(32), ToTensor()]))
testloader = DataLoader(transformed_dataset_test, batch_size=batch_size,
                        shuffle=True, pin_memory=True)

if isGhostNet:
    netType = "ghostNet_{}_{}".format(args.ratio1,args.ratio2)
    model = GhostNet(args.ratio1,args.ratio2)

else:
    netType="ecladNet"
    model = Net()

    
model.load_state_dict(torch.load('runs/model/pytorch/{}.pt'.format(netType)))
model.eval()
model.to(device)

# Profile path
profile_path = "pytorchModels/profiles/"
if not(os.path.exists(profile_path)):
    os.makedirs(profile_path)

with torch.no_grad():
    for i, sample_batched in enumerate(testloader):
        inputs = sample_batched['image'].to(device)
        if i == 0:
            output = model(inputs)
            # Profile second and third batch as first batch initiate cuda kernels.
        else:
            with torch.profiler.profile(
                with_stack=True,
                profile_memory=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('pytorchModels/profiles/{}'.format(netType))
            )as prof:
                output = model(inputs)
        if i == 1 :
            break
    



