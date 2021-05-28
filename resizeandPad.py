import cv2
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob
import argparse
import math
import sys


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        print(f"height ={height} and h = {float(h)} and r = {r} ")
        dim = (int(w * r), height)
        print(f"height ={height} and dim = {dim}")
        

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        print(f"width ={width} and w = {float(w)} and r = {r} ")
        dim = (width, int(h * r))
        print(f"width ={width} and dim = {dim}")
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized



def drawBox(boxes, image):
    print(boxes)
    for i in range(0, len(boxes)):
        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()


parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--folder', help='Path to video file.')
args = parser.parse_args()

folderPath = r"/home/transferlearningtoolkit/tlt-experiments/Data/Data/differentres/training"

if (args.folder):
    # Open the image file
    if not os.path.isdir(args.folder):
        print("Directory   ", args.folder, " doesn't exist")
        sys.exit(1)
    
    folderPath = args.folder

    # cap = cv.VideoCapture(args.image)


images = os.path.join(folderPath, "image_2")
labels = os.path.join(folderPath, "label_2")
convertedimages = os.path.join(folderPath, "convertedimage_2")
convertedlabel = os.path.join(folderPath, "convertedlabel_2")
if not os.path.exists(convertedimages):
    os.makedirs(convertedimages)
if not os.path.exists(convertedlabel):
    os.makedirs(convertedlabel)


imageFiles = os.listdir(images)
labelFiles = os.listdir(labels)
print(images)
print(labels)


ratio_output= (640,480)
color = [0, 0, 0] # 'cause purple!
for image in imageFiles:
    textfile = image[:-4]+".txt"
    if  textfile not in labelFiles:
        print(labels)
        print(image[:-4])
        continue
    imageName = os.path.join(images,image)
    print(imageName)
    labelFile = os.path.join(labels,textfile)
    convertedimageFile = os.path.join(convertedimages,image)
    convertedlabelFile = os.path.join(convertedlabel,textfile)
    print(imageName)
    originalImage = cv2.imread(imageName)
    print(originalImage.shape)

    img = image_resize(originalImage,height=ratio_output[1])
    right= ratio_output[0] - img.shape[1] 
    print(right)

    xScalePaddedwithRatio = img.shape[1]  /  originalImage.shape[1]
    yScalePaddedwithRatio =  img.shape[0] /  originalImage.shape[0]

    if right < 0 :
        imgPadded = cv2.resize(img, ratio_output);
        xScalePaddedwithRatio = imgPadded.shape[1]  /  originalImage.shape[1]
        yScalePaddedwithRatio =   imgPadded.shape[0] /  originalImage.shape[0]
    else:
        imgPadded = cv2.copyMakeBorder(img,0,0, 0, right, cv2.BORDER_CONSTANT, value=color)
        xScalePaddedwithRatio = img.shape[1]  /  originalImage.shape[1]
        yScalePaddedwithRatio =  img.shape[0] /  originalImage.shape[0]


    plt.imshow(imgPadded[:,:,::-1])  
    img = np.array(imgPadded)
    file = open(convertedlabelFile, 'w')
    cv2.imwrite(convertedimageFile,img)
    # plt.imsave("onePic.jpg", originalImage)
    with open(labelFile, 'r') as reader:
        line = reader.readline()
        bboxes=[]
        while line != '':  # The EOF char is an empty string
            lists=line.split( )
            name = lists[0]
            originalbbox =lists[4:8]
            bboxInt = [int(float(i)) for i in originalbbox]
            finalx = int(np.round(bboxInt[0] * xScalePaddedwithRatio))
            finaly = int(np.round(bboxInt[1] * yScalePaddedwithRatio))
            finalxmax = int(np.round(bboxInt[2] * xScalePaddedwithRatio))
            finalymax = int(np.round(bboxInt[3] * yScalePaddedwithRatio))
            box=[finalx,finaly,finalxmax, finalymax]
            bboxes.append(box)
            writte =  f"{name} 0.00 0 0.0 {finalx}.00 {finaly}.00 {finalxmax}.00 {finalymax}.00 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
            file.write(writte)
            line = reader.readline()
            if line !='':
                file.write("\n")
        drawBox(bboxes, img)
    file.close()
            


