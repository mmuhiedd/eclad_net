import os
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("{cur_dir}/../pytorchModels/".format(cur_dir=os.path.dirname(os.path.abspath(__file__))))
from definitions import *
from classes import *
import timeit


def load_batch_image(batch_size=32):
    img_list_test, class_list_test_t = read_images(train=False)
    class_list_test_t = onehotencoding_class(class_list_test_t)
    transformed_dataset_test = Logo_Dataset(img_list_test, class_list_test_t,
                                         transform = transforms.Compose([Rescale(32), ToTensor()]))
    testloader = DataLoader(transformed_dataset_test, batch_size=batch_size,
                        shuffle=True, pin_memory=True)
    # Retrieve only first batch
    for _,batch in enumerate(testloader):
        images_batch = batch["image"].numpy()
        labels_batch = batch["class_name"].numpy()
        return images_batch,labels_batch
    

def compute_inference_time(module):
    '''
    By default it's only on one batch
    '''

    timing_number = 10
    timing_repeat = 10
    metrics = (
        np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
        * 1000
        / timing_number
    )
    metrics = {
        "mean": np.mean(metrics),
        "median": np.median(metrics),
        "std": np.std(metrics),
    }

    return metrics

def compute_acc(preds,labels):
    correct,total = 0.0,0.0

    for cur_pred,label in zip(preds,labels):
        cur_max_pred_id = np.argmax(cur_pred)
        if cur_max_pred_id == label:
            correct += 1
        total +=1
            

    acc = correct/total
    return acc, correct, total