import sys
from tvm.driver import tvmc
import tvm
import argparse
import os
import onnx
from tvm.contrib import graph_executor
import numpy as np
from TVMdefinitions import load_batch_image

##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule',type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule',type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)
args = parser.parse_args()
isGhostNet = args.ghost


#### Load TestSet
# Shape of one raw images : [3,30,30]
# Shape required by the model : [32,3,32,32]
images_arr, labels_arr = load_batch_image()

#### Load the ONNX model
onnx_path = "runs/model/onnx"
onnx_file = os.path.join(onnx_path,"ghostEclad_{}_{}.onnx".format(args.ratio1,args.ratio2)) if isGhostNet else os.path.join(onnx_path,"ecladNet.onnx")
onnx_model = onnx.load(onnx_file)


#### Compile Model with Relay
target = "llvm"
input_name = "input"
print(images_arr.shape)
shape_dict = {input_name: images_arr.shape}

mod, params = tvm.relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=3):
  lib = tvm.relay.build(mod, target=target, params=params)


device = tvm.device(str(target), 0)
print(lib["default"](device))

module = graph_executor.GraphModule(lib["default"](device))


### Execute on TVM Runtime 
dtype = "float32"
module.set_input(input_name, images_arr)
module.run()
output_shape = (32, 2)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
