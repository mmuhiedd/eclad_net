import sys
import tvm
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
import tvm.relay as relay
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
parser.add_argument('--no_tune',help='set to not tune. useful tuning already performed',action='store_true', required=False, default=False)

args = parser.parse_args()
isGhostNet = args.ghost
isTuning = not(args.no_tune)


printTask = False



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

mod, params = relay.frontend.from_onnx(onnx_model)
""" example of mod
def @main(%input: Tensor[(32, 3, 32, 32), float32], %fc1.weight: Tensor[(32, 600), float32], %fc1.bias: Tensor[(32),.... {
  %0 = nn.conv2d(%input, %onnx::Conv_34, padding=[0, 0, 0, 0], channels=12, kernel_size=[5, 5]);
  %1 = nn.bias_add(%0, %onnx::Conv_35);
  %2 = nn.relu(%1);
  %3 = nn.max_pool2d(%2, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0]);
  %4 = nn.conv2d(%3, %onnx::Conv_37, padding=[0, 0, 0, 0], channels=24, kernel_size=[5, 5]);
.....
}
 """


with tvm.transform.PassContext(opt_level=3):
  lib = relay.build(mod, target=target, params=params)


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
acc, correct, total = compute_acc(tvm_output,labels_arr)
print("Accuracy : {}  ({}/{})".format(acc,correct,total))


'''
Tuning !! 
'''

# tuning path path
tuning_path = "TVM/tuning/"
if not(os.path.exists(tuning_path)):
    os.makedirs(tuning_path)
tuning_file =os.path.join(tuning_path,"ghostEclad_{}_{}_tuning.json".format(args.ratio1,args.ratio2)) if isGhostNet else os.path.join(tuning_path,"ecladNet_tuning.json")



'''
RUNNER TVM
The runner takes compiled code that is generated with a specific set of parameters and measures the performance of it.
'''
# The number of times to run the generated code for taking average. We call these runs as one repeat of measurement.
number = 20

# The number of times to repeat the measurement. In total, the generated code will be run (1 + number x repeat) times,
# Here then, 20 run for 1 measurement => 20x3 run for 3 measurement. This is done for all 'trials' configuration wich is set here to 
# 2000. Then, for each task, 2000x20x3 run total. (not take into account 'min_repeat_ms' for clarity).
repeat = 3
# specifies how long need to run configuration test. If the number of repeats falls under this time, it will be increased. 
min_repeat_ms = 4
# upper limit on how long to run training code for each tested configuration.
timeout = 10  # in seconds
# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms
)

# Trials : number of different configurations 
tuning_option = {
    "tuner": "xgb",
    "trials": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"),
        runner=runner
    ),
    "tuning_records": tuning_file,
}

# begin by extracting the tasks from the onnx model
# Correspond to all layer of the model.
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

if printTask:
    for _,task in enumerate(tasks):
        print("\n\n %s" % task)

# Tune the extracted tasks sequentially.
if isTuning:
  print("Start TUNING\n")
  for i, task in enumerate(tasks):
      prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
      tuner_obj = XGBTuner(task, loss_type="rank")
      tuner_obj.tune(
          n_trial=min(tuning_option["trials"], len(task.config_space)),
          early_stopping=tuning_option["early_stopping"],
          measure_option=tuning_option["measure_option"],
          callbacks=[
              autotvm.callback.progress_bar(min(tuning_option["trials"], len(task.config_space)),
                                             prefix=prefix),
              autotvm.callback.log_to_file(tuning_option["tuning_records"]),
          ],
      )
  print("End TUNING\n")


  ## Test best tuned model
  with autotvm.apply_history_best(tuning_option["tuning_records"]):
      with tvm.transform.PassContext(opt_level=3, config={}):
          lib = tvm.relay.build(mod, target=target, params=params)

  dev = tvm.device(str(target), 0)
  tuned_module = graph_executor.GraphModule(lib["default"](dev))

  dtype = "float32"
  tuned_module.set_input(input_name, images_arr)
  tuned_module.run()
  output_shape = (32, 2)
  tvm_output = tuned_module.get_output(0, tvm.nd.empty(output_shape)).numpy()

  ## Exec times of tuned model
  tuned_metrics = compute_inference_time(tuned_module)
  # time in ms
  print("untuned: ")
  print(metrics)
  print("\n tuned: ")
  print(tuned_metrics)
  ## Accuracy score
  tuned_acc, _, _ = compute_acc(tvm_output,labels_arr)
  print("untuned :  Accuracy : {}  ({}/{})".format(acc,correct,total))
  print("Tuned :    Accuracy : {}  ({}/{})".format(tuned_acc,correct,total))

