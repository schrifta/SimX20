# This scripts splits images and label files in given dataset directory
# that contains a complete directory tree.
# It searches for 'train', 'test' and 'valid' subdirectories, and within each it
# searches for 'images', and 'labels'.
# It splits images and labels files to 2x2 and saves the results in a given output
# dataset directory.

# Import the opencv library
import cv2
import os
import copy
import numpy as np


# Define a label class
class Label(object):
    def __init__(self, cls, x, y, w, h):
        self.cls = cls
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# Define a function to split the label files
def split_annotations(labels, x1, x2, y1, y2):
    lt = []
    rt = []
    lb = []
    rb = []

    for lbl in labels:
        xmin = lbl.x - lbl.w/2
        xmax = xmin + lbl.w
        ymin = lbl.y - lbl.h/2
        ymax = ymin +lbl.h
        # left top
        if (xmin<x2 or xmax<x1) and (ymin<y2 or ymax<y1):
            # lt.append(Label(lbl.cls, lbl.x / x1, lbl.y / y1, lbl.w / x1, lbl.h / y1))
            nl = Label(lbl.cls, lbl.x / x1, lbl.y / y1, lbl.w / x1, lbl.h / y1)
            if xmax>x1:
                nl.w = 1 - (nl.x-nl.w/2)
                nl.x = (1 + nl.w)/2
            if ymax>y1:
                nl.h = 1 - (nl.y-nl.h/2)
                nl.y = (1 + nl.h)/2
            lt.append(nl)
        # right top
        if (xmax>x1 or xmin>x2) and (ymin<y2 or ymax<y1):
            # rt.append(Label(lbl.cls, (lbl.x - x2) / x1, lbl.y / y1, lbl.w / x1, lbl.h / y1))
            nl = Label(lbl.cls, (lbl.x - x2) / x1, lbl.y / y1, lbl.w / x1, lbl.h / y1)
            if xmin<x2:
                nl.w = nl.x + nl.w/2
                nl.x = nl.w/2
            if ymax>y1:
                nl.h = 1 - (nl.y-nl.h/2)
                nl.y = (1 + nl.h)/2
            rt.append(nl)
        # left bottom
        if (xmin<x2 or xmax<x1) and (ymax>y1 or ymin>y2):
            # lb.append(Label(lbl.cls, lbl.x / x1, (lbl.y - y2) / y1, lbl.w / x1, lbl.h / y1))
            nl = Label(lbl.cls, lbl.x / x1, (lbl.y - y2) / y1, lbl.w / x1, lbl.h / y1)
            if xmax>x1:
                nl.w = 1 - (nl.x-nl.w/2)
                nl.x = (1 + nl.w)/2
            if ymin<y2:
                nl.h = nl.y + nl.h/2
                nl.y = nl.h/2
            lb.append(nl)
        # right bottom
        if (xmax>x1 or xmin>x2) and (ymax>y1 or ymin>y2):
            # rb.append(Label(lbl.cls, (lbl.x - x2) / x1, (lbl.y - y2) / y1, lbl.w / x1, lbl.h / y1))
            nl = Label(lbl.cls, (lbl.x - x2) / x1, (lbl.y - y2) / y1, lbl.w / x1, lbl.h / y1)
            if xmin < x2:
                nl.w = nl.x + nl.w / 2
                nl.x = nl.w / 2
            if ymin < y2:
                nl.h = nl.y + nl.h / 2
                nl.y = nl.h / 2
            rb.append(nl)
    return lt, rt, lb, rb

# Define a function to resize the label files
def ResizeAnnotations(labels, x0, y0):
    newLabels = []
    for lbl in labels:
        nl = Label(lbl.cls, x0+lbl.x/2, y0+lbl.y/2, lbl.w/2, lbl.h/2)
        newLabels.append(nl)
    return newLabels


# Define the dataset directories
inputDir = "D:/CytobitData/DataSets/My/Spr#12-3-my"
outputDir = "D:/CytobitData/DataSets/My/Spr#12-3-my/X20-single"

try:
    os.mkdir(outputDir)
except OSError as error:
    if os.path.isdir(outputDir):
        print(outputDir + " already exists!")
    else:
        print(outputDir + " creation problem!")
        exit(-1)

showAnnotations = False

# Loop through the subdirectories
subdirs = ["train", "test", "valid"]
for subdir in subdirs:

    # Check for input image directory
    inputImageDir = inputDir + "/" + subdir + "/images"
    if not os.path.isdir(inputImageDir):
        print(inputImageDir + " missing!")
        exit(-1)

    # Check for input label directory
    inputLabelDir = inputDir + "/" + subdir + "/labels"
    if not os.path.isdir(inputLabelDir):
        print(inputLabelDir + " missing!")
        exit(-1)

    # Create subdir
    outputSubdir = outputDir + "/" + subdir
    try:
        os.mkdir(outputSubdir)
    except:
        if os.path.isdir(outputSubdir):
            print(outputSubdir + " already exists!")
        else:
            print(outputSubdir + " creation problem!")
            exit(-1)

    # Create image output directory
    outputImageDir = outputSubdir + "/images"
    try:
        os.mkdir(outputImageDir)
    except OSError as error:
        if os.path.isdir(outputImageDir):
            print(outputImageDir + " already exists!")
        else:
            print(outputImageDir + " creation problem!")
            exit(-1)

    # Create Label output directory
    outputLabelDir = outputSubdir + "/labels"
    try:
        os.mkdir(outputLabelDir)
    except OSError as error:
        if os.path.isdir(outputLabelDir):
            print(outputLabelDir + " already exists!")
        else:
            print(outputLabelDir + " creation problem!")
            exit(-1)

    # Loop through the files in the input image directory
    idx = 0
    newImage = []
    newLabels = []
    imageNamesStr = []
    for filename in os.listdir(inputImageDir):
        # Get file extension
        if len(filename) > 4 and not os.path.isdir(filename) and filename[-4] == '.':
            ext = filename[-4:]
            if ext in {'.jpg', '.png', '.bmp'}:
                # Read the input image
                image = cv2.imread(os.path.join(inputImageDir, filename))
                height, width = image.shape[:2]

                # Read image annotations
                basename = filename[:-4]
                Label_file_name = os.path.join(inputLabelDir, basename + ".txt")
                labels = []
                with open(Label_file_name, "r") as f:
                    for line in f:
                        cls, x, y, w, h = line.split()
                        x = float(x)
                        y = float(y)
                        w = float(w)
                        h = float(h)
                        labels.append(Label(cls, x, y, w, h))

                # Paste the resized input image to the output image
                x0 = int(width/4)
                y0 = int(height/4)
                resizedImage = cv2.resize(image,(int(width/2),int(height/2)))
                newImage = np.zeros((height, width,3), np.uint8)
                newLabels = []
                imageNamesStr = '[' + basename + ']'
                    
                newImage[y0:y0+int(height/2),x0:x0+int(width/2)] = resizedImage

                # Resize and append the annotations
                resizedLabels = ResizeAnnotations(labels,x0/width,y0/height)
                for lbl in resizedLabels:
                    newLabels.append(lbl)
                                        
                # Save the resized files
                if len(newLabels) > 0:
                    cv2.imwrite(os.path.join(outputImageDir, basename + "_comb" + ext), newImage)
                    with open(os.path.join(outputLabelDir, basename + "_comb.txt"), "w") as f:
                        for lbl in newLabels:
                            f.write('%s %8.6f %8.6f %8.6f %8.6f\n' % (lbl.cls, lbl.x, lbl.y, lbl.w, lbl.h))

                # Show the images with annotations
                if showAnnotations:
                    for lbl in newLabels:
                        #color = (0,200,0) if lbl.cls[0]=='0' else (0,200,200)  # YOLOv8Clip.exe colors
                        color = (0,255,0) if lbl.cls=='1' else (0,80,255) # TestDataset.py colors
                        cv2.rectangle(newImage,
                                      (int(width * (lbl.x - lbl.w / 2)), int(height * (lbl.y - lbl.h / 2))),
                                      (int(width * (lbl.x + lbl.w / 2)), int(height * (lbl.y + lbl.h / 2))),
                                      color, 2)
                    cv2.imshow(imageNamesStr, newImage)
                    if 27 == int(cv2.waitKeyEx(0)):
                        cv2.destroyAllWindows()
                        exit(0)


                # Set the index within an image quadruple
                cv2.destroyAllWindows()
                idx = (idx+1)%4