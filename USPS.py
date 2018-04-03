#Extracting file from USPS dataset
import gzip, zipfile
import sys, os
import numpy as np
from PIL import Image

filename="proj3_images.zip"

#Defining height,width for resizing the images to 28x28 like MNIST digits
height=28
width=28

#Defining path for extracting dataset zip file
extract_path = "usps_data"

#Defining image,label list
images = []
img_list = []
labels = []

#Extracting given dataset file    
with zipfile.ZipFile(filename, 'r') as zip:
    zip.extractall(extract_path)

#Extracting labels,images array needed for training    
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
        
    if "Numerals" in path:
        image_files = [fname for fname in files if fname.find(".png") >= 0]
        for file in image_files:
            labels.append(int(path[-1]))
            images.append(os.path.join(*path, file)) 

#Resizing images like MNIST dataset   
for idx, imgs in enumerate(images):
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_list.append(img_data)

#Storing image and labels in arrays to be used for training   
USPS_img_array = np.array(img_list)
USPS_img_array = np.subtract(255, USPS_img_array)
USPS_label_array = np.array(labels)

#printing
#print(USPS_img_array)
#print(USPS_label_array)
