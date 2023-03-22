from PIL import Image
import os
import numpy as np

def load_batch_image(batch_size=32):
    '''
    Load 'batch_size' images and corresponding labels from the test set
    '''
    src = './dataset/test'

    half_batch_three = batch_size//2
    
    img_array = np.zeros(shape=(batch_size, 3, 32, 32))
    class_array = np.zeros(shape=(batch_size))
    i = 0

    for dirpath, dirs,_ in os.walk(src):
        for d in dirs:
            dir_str = os.path.join(dirpath, d)
            
            # define class hot encoding
            if d =="res": 
                class_ohe = 0
            else:
                class_ohe = 1

            for _, _, files in os.walk(dir_str):
                for j, file in enumerate(files):
                    temp_img_str = os.path.join(dir_str, file)
                    resized_image = Image.open(temp_img_str).resize((32,32))

                    # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
                    resized_image = np.transpose(resized_image, (2, 0, 1))

                    img_array[i*half_batch_three + j] = np.array(resized_image).astype("float32")
                    class_array[i*half_batch_three + j] = class_ohe

                    if j == half_batch_three-1:
                        i += 1 
                        break

    return img_array, class_array