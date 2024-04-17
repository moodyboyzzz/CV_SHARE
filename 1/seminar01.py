import numpy as np
from PIL import Image
import pandas as pd

image = np.array(Image.open('image.png')) / 255.0

averaged_image = np.mean(image, axis=2)

task = pd.read_csv('task.csv')
row = task['row'].values[0]
col = task['col'].values[0]

len = task['len'].values[0]
patch = averaged_image[row:row+len, col:col+len]

np.save('seminar01_crop.npy', patch, allow_pickle=False)