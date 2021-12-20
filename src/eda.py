import pandas as pd
import os
import matplotlib.pyplot as plt
from imageio import imread

train_df = pd.read_csv('Project_data/train.csv',names= ['folder','class_name','label'],sep=';')
print(train_df.label.unique())
for label in train_df.label.unique():
    folder = train_df[train_df.label==label].iloc[0,0]
    fig, axes = plt.subplots(6,5, figsize=(36,30))
    for id, files in enumerate(zip(sorted(os.listdir(os.path.join(*['Project_data','train',folder]))),axes.flatten())):
        filename, ax = files
        print(filename)
        ax.imshow(imread(os.path.join(*['Project_data','train',folder,filename])))
        ax.title.set_text(f'{label}_{id+1}')
        
    plt.show()
        

print(train_df.head())