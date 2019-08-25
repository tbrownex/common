import pathlib
import os

def mapImageToClass(config):
    ''' 
    The intent is to map a file name to a Class, particulary for image processing.
    The folder structure is assumed to be:
      - root
        - class1
        - class2
        - ...
        - classN
        
        where each "class" folder contains images of that class
        
    'imageClass' will have entries like <image001.jpg>: 'tulip'
    'classIdx' will have entries like <tulip>: 3
    '''
    root = config["dataLoc"]
    
    classIdx = {}
    imageClass = {}
    idx = 0
    
    for root, dirs, files in os.walk(root):
        for cls in dirs:
            classIdx[cls] = idx
            idx += 1
            for r, d, files in os.walk(root+cls):
                for f in files:
                    fullPath = r+"/"+f
                    imageClass[fullPath] = cls
    return imageClass, classIdx