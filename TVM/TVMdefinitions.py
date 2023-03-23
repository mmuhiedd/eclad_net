import os
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("{cur_dir}/../pytorchModels/".format(cur_dir=os.path.dirname(os.path.abspath(__file__))))
from definitions import *
from classes import *


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
