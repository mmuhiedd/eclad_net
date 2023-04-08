import torch
import argparse
import sys
import os
sys.path.append("{cur_dir}/../pytorchModels/".format(cur_dir=os.path.dirname(os.path.abspath(__file__))))
from classes import GhostNet,Net
from definitions import *
import os





##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule. Default = 2', type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule. Default = 2', type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)
parser.add_argument('--batch',help='Batch Size used in ONNX. Default = 32', type=int, default=32)



args = parser.parse_args()

isGhostNet = args.ghost
batch_size = args.batch

# An example input you would normally provide to your model's forward() method.
dummy_input = torch.rand(batch_size, 3, 32, 32)

onnx_path = "runs/model/onnx"
if not(os.path.exists(onnx_path)):
    os.makedirs(onnx_path)

if isGhostNet:
    netType = "GhostNet"
    model = GhostNet(args.ratio1,args.ratio2)
    model.load_state_dict(torch.load('runs/model/pytorch/ghostNet_{}_{}.pt'.format(args.ratio1,args.ratio2)))
    model.eval()
    torch_out = model(dummy_input)
    # Export to ONNX
    torch.onnx.export(model, dummy_input, os.path.join(onnx_path,"ghostEclad_{}_{}_b{}.onnx".format(args.ratio1,args.ratio2,batch_size)),
                       export_params=True,
                         input_names = ['input'])
    

else:
    netType="ECLAD-Net"
    model = Net()
    model.load_state_dict(torch.load('runs/model/pytorch/ecladNet.pt'))
    model.eval()
    torch_out = model(dummy_input)
    # Export to ONNX
    torch.onnx.export(model, dummy_input, os.path.join(onnx_path,"ecladNet_b%d.onnx" % batch_size),
                    export_params=True,
                    input_names = ['input'])







