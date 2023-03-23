import sys
from tvm.driver import tvmc
import tvm
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import argparse
import os
import onnx
from tvm.contrib import graph_executor
import numpy as np
from TVMdefinitions import * 

##############
## ARGS PARSING 
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ratio1',help='ratio of ghost for first ghostmodule',type=int, default=2)
parser.add_argument('--ratio2',help='ratio of ghost for second ghostmodule',type=int, default=2)
parser.add_argument('--ghost',help='process ghost net',action='store_true', required=False, default=False)
parser.add_argument('--arch',help='arch targetted. Default is llvm',type=str, default="llvm")
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
target = args.arch
input_name = "input"
shape_dict = {input_name: images_arr.shape}

mod, params = tvm.relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=3):
  lib = tvm.relay.build(mod, target=target, params=params)


device = tvm.device(str(target), 0)

module = graph_executor.GraphModule(lib["default"](device))


### Execute on TVM Runtime 
dtype = "float32"
module.set_input(input_name, images_arr)
module.run()
output_shape = (32, 2)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()


""" 
PERFORMANCES
 """

## Exec times 
metrics = compute_inference_time(module)
# time in ms
print(metrics)



## Accuracy score
compute_acc(tvm_output,labels_arr)



'''
Tuning !! 
'''
# tuning path path
tuning_path = "TVM/tuning/"
if not(os.path.exists(tuning_path)):
    os.makedirs(tuning_path)
tuning_file =os.path.join(tuning_path,"ghostEclad_{}_{}_tuning.json".format(args.ratio1,args.ratio2)) if isGhostNet else os.path.join(tuning_path,"ecladNet_tuning.json")


number = 10
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": tuning_file,
}

# begin by extracting the tasks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )