import sys
import os
import argparse
import numpy as np
import cv2
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math
from skimage.exposure import rescale_intensity

# My designed Convolution function
def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                    roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                    k = (roi * kernel).sum()
                    output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output



# My gaussian(box) template used for filtering. 

boxfilter = np.array((
	[1/9, 1/9, 1/9],
	[1/9, 1/9, 1/9],
	[1/9, 1/9, 1/9]))



# downsampling function

def downsample(img):
  out = None
  kernel = boxfilter
  convolveOutput1 = convolve(img, boxfilter)
  out = convolveOutput1[::2,::2]
  return out

 
# upsampling function

def upsample(image):
  out = None
  kernel = boxfilter
  outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
  outimage[::2,::2]=image[:,:]
  convolveOutput1 = convolve(outimage, boxfilter)
  
  out = 4*convolveOutput1
  return out




# Creating the Gaussian Pyramid for a given image 

def gauss_pyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0,levels):
    tmp = downsample(tmp)
    output.append(tmp)
  return output
 



# Creating the Laplacian Pyramid for a given image using the generated gaussian pyramids

def lapl_pyramid(gauss_pyr):
  output = []
  k = len(gauss_pyr)
  for i in range(0,k-1):
    gu = gauss_pyr[i]
    egu = upsample(gauss_pyr[i+1])
    if egu.shape[0] > gu.shape[0]:
       egu = np.delete(egu,(-1),axis=0)
    if egu.shape[1] > gu.shape[1]:
      egu = np.delete(egu,(-1),axis=1)
    output.append(gu - egu)
  output.append(gauss_pyr.pop())
  return output


# Blend- Function to blend the LApalcian pyramids in accordance with the mask

def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
  blended_pyr = []
  k= len(gauss_pyr_mask)
  for i in range(0,k):
   p1= gauss_pyr_mask[i]*lapl_pyr_white[i]
   p2=(1 - gauss_pyr_mask[i])*lapl_pyr_black[i]
   blended_pyr.append(p1 + p2)
  return blended_pyr



#Collapse - Function for  Image reconstruction based on the laplacian pyrmaid levels

def collapse(lapl_pyr):
  output = None
  output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = upsample(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    lapl_pyr.pop()
    lapl_pyr.pop()
    lapl_pyr.append(tmp)
    output = tmp
  return output



# Rectangular GUI Code
print("Drag from the top left corner to the bottom right corner, to obtain the rectangular region for the mask") 

ref_point = [] 
crop = False
  
def shape_selection(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point, crop 
  
    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)] 
  
    # check to see if the left mouse button was released 
    elif event == cv2.EVENT_LBUTTONUP: 
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
        ref_point.append((x, y)) 
  
        # draw a rectangle around the region of interest 
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)


        cv2.rectangle(bwmask, ref_point[0], ref_point[1], (255, 255, 255), -1)

        
        cv2.imshow("image", image)
        cv2.imshow("mask of the image",bwmask)
  
  
  
# load the image, clone it, and setup the mouse callback function 
image=cv2.imread("dt3.png")
bwmask = np.zeros(shape = image.shape, dtype = "uint8")
#image = cv2.imread(args["image"]) 
clone = image.copy() 
cv2.namedWindow("image") 
cv2.setMouseCallback("image", shape_selection) 
  
  
# keep looping until the 'q' key is pressed 
while True: 
    # display the image and wait for a keypress 
    cv2.imshow("image", image) 
    key = cv2.waitKey(1) & 0xFF
  
    # press 'r' to reset the window 
    if key == ord("r"): 
        image = clone.copy() 
  
    # if the 'c' key is pressed, break from the loop 
    elif key == ord("c"): 
        break
  

# Main Function starts from here; as all the defined functions are above


 

image1 = cv2.imread('ml_rs.png') # Background image
image2 = cv2.imread('dt3.png')   # Foreground image
mask = bwmask                    # Black and white mask used for blending
(b1,g1,r1)=cv2.split(image1)
(b2,g2,r2)=cv2.split(image2)
(bm,gm,rm)=cv2.split(mask)


r1 = r1.astype(float)
g1 = g1.astype(float)
b1 = b1.astype(float)

r2 = r2.astype(float)
g2 = g2.astype(float)
b2 = b2.astype(float)

rm = rm.astype(float)/255
gm = gm.astype(float)/255
bm = bm.astype(float)/255


# Figuring out the number of levels to implement the Gaussian Pyramid
rows=image1.shape[0]
cols=image1.shape[1]
area=rows*cols

depth = int(math.floor(math.log(area, 2))-1) #  4x4 image at the most downsampled level.
print('The number of levels (depth) of the gaussian pyramid=',depth)

### Automatically figure out the size
##min_size = min(r1.shape)
###depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.
##depth = int(math.floor(math.log(min_size, 2))) - 1 # at least 4x4 at the highest level.


gauss_pyr_maskr = gauss_pyramid(rm, depth)
gauss_pyr_maskg = gauss_pyramid(gm, depth)
gauss_pyr_maskb = gauss_pyramid(bm, depth)

gauss_pyr_image1r = gauss_pyramid(r1, depth)
gauss_pyr_image1g = gauss_pyramid(g1, depth)
gauss_pyr_image1b = gauss_pyramid(b1, depth)

gauss_pyr_image2r = gauss_pyramid(r2, depth)
gauss_pyr_image2g = gauss_pyramid(g2, depth)
gauss_pyr_image2b = gauss_pyramid(b2, depth)

lapl_pyr_image1r  = lapl_pyramid(gauss_pyr_image1r)
lapl_pyr_image1g  = lapl_pyramid(gauss_pyr_image1g)
lapl_pyr_image1b  = lapl_pyramid(gauss_pyr_image1b)

lapl_pyr_image2r = lapl_pyramid(gauss_pyr_image2r)
lapl_pyr_image2g = lapl_pyramid(gauss_pyr_image2g)
lapl_pyr_image2b = lapl_pyramid(gauss_pyr_image2b)

outpyrr = blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr)
outpyrg = blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg)
outpyrb = blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb)

outimgr = collapse(blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr))
outimgg = collapse(blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg))
outimgb = collapse(blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb))


# Removing the outliers that sometimes occur due to blending
outimgr[outimgr < 0] = 0
outimgr[outimgr > 255] = 255
outimgr = outimgr.astype(np.uint8)

outimgg[outimgg < 0] = 0
outimgg[outimgg > 255] = 255
outimgg = outimgg.astype(np.uint8)

outimgb[outimgb < 0] = 0
outimgb[outimgb > 255] = 255
outimgb = outimgb.astype(np.uint8)

result = np.zeros(image1.shape,dtype=image1.dtype)
tmp = []
tmp.append(outimgb)
tmp.append(outimgg)
tmp.append(outimgr)
result = cv2.merge(tmp,result)
cv2.imshow("blended image",result) # final merged image
 
