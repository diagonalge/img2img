import cv2
from PIL import Image, ImageOps
import numpy as np
from math import floor, ceil

def resize(image):
    landscape = False
    ratio_dict = {1: 512, 0.56: 648, 0.75: 512, 1.5: 648}
    height = int(image.size[0])
    width = int(image.size[1])
    if int(height) > int(width):
        ratio = width/height
    else:
        ratio = height/width
        landscape = True
    nearest_key = min(ratio_dict, key=lambda x: abs(x - ratio))
    nearest_value = ratio_dict[nearest_key]
    if height > width:
        bigger_side = height
    else:
        bigger_side = width

    bigger_side = nearest_value
    smaller_side = int(bigger_side*ratio)
    resized = resize_img(image,bigger_side,smaller_side, landscape)
    resized = crop_by_eight(resized)
    return resized


def resize_img(image,bigger_side,smaller_side, landscape):
    output = image
    dsize = (smaller_side, bigger_side) if landscape else (bigger_side, smaller_side)
    output = image.resize(dsize, Image.Resampling.LANCZOS)
    return output

def crop_by_eight(image):
    height = image.size[0]
    width = image.size[1]
    x, a, b, crop_value = 0,0,0,0
    if height%8!=0:
        x = closestNumber(height)
        crop_value = height - x
        if (crop_value%2==0):
            a = crop_value/2
            b = a
        else:
            a = floor(crop_value/2)
            b = ceil(crop_value/2)
        border = (a,0,b,0)
        image = ImageOps.crop(image, border)
    if width%8!=0:
        x = closestNumber(width)
        crop_value = width - x
        if (crop_value%2==0):
            a = crop_value/2
            b = a
        else:
            a = floor(crop_value/2)
            b = ceil(crop_value/2)
        border = (0,a,0,b)
        image = ImageOps.crop(image, border)

    return image
    


def closestNumber(n) :
    while(n%8!=0):
        n-=1
    return n