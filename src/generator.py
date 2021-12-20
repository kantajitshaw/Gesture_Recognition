import numpy as np
import os
# from scipy.misc import imread, imresize
from PIL import Image
import datetime
import os
import matplotlib.pyplot as plt

np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.random.set_seed(30)
import math

train_doc = np.random.permutation(open('Project_data/train.csv').readlines())
val_doc = np.random.permutation(open('Project_data/val.csv').readlines())
batch_size = 2 #experiment with the batch size

def generator(source_path, folder_list, batch_size, input_shape=(224,224), ablation=None, verbose=0):
    print( 'Source path = ', source_path, '; batch size =', batch_size, '; input_shape =', input_shape, ';ablation =',ablation)
    img_idx = list(range(4,30,5)) #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        if ablation!=None:
            new_folder_list=[]
            classes = set([int(line.split(';')[2]) for line in t])
            for label in classes:
                new_folder_list.extend([line for line in t if int(line.split(';')[2])==label][:ablation])
                
            t = np.random.permutation(new_folder_list)
                
        num_batches = int(math.ceil(len(t)/batch_size))# calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            cur_batch_size = batch_size
            if len(t)%batch_size!=0 and batch==num_batches-1:
                cur_batch_size = len(t)%batch_size
            batch_data = np.zeros((cur_batch_size,len(img_idx),input_shape[0],input_shape[1],3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((cur_batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(cur_batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = Image.open(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item])
                    image = image.resize(input_shape)
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    image = np.asarray(image).astype(np.float32)
                    
                    batch_data[folder,idx,:,:,0] = image[:,:,0]/255
                    batch_data[folder,idx,:,:,1] = image[:,:,1]/255
                    batch_data[folder,idx,:,:,2] = image[:,:,2]/255
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches


curr_dt_time = datetime.datetime.now()
train_path = 'Project_data/train'
val_path = 'Project_data/val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 10 # choose the number of epochs
print ('# epochs =', num_epochs)

fig, axes = plt.subplots(4, 4, figsize=(30,30))
axes = axes.flatten()
fig.tight_layout()
test_generator =  generator(train_path, train_doc, batch_size, ablation=10)
for id in range(16):
    data, label = next(test_generator)
    print(data.shape, label.shape)
    axes[id].imshow(data[0,0,:,:,:])
    axes[id].title.set_text(str(np.argmax(label[0])))
    
plt.show()