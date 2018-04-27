# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:07:40 2018

@author: T901
"""

#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import skimage.color as sc
import urllib.request

print("\n1. Load an image")
#!curl https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg -o img.jpg
urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg', 'img.jpg')

i = np.array(Image.open('img.jpg'))
imshow(i)
#plt.axis('off')
plt.show()

print("\n2. Examine Numerical Properties")
print('Color image: ', 'type = ', type(i), ', data type = ', i.dtype, ', shape = ', i.shape)

i_mono = sc.rgb2gray(i)   # Y = 0.2125 R + 0.7154 G + 0.0721 B
imshow(i_mono, cmap='gray')
#plt.axis('off')
#plt.show()
print('Gray image: shape = ', i_mono.shape) 

print("\n3. View Pixel Value Distributions")
print("Plot a histogram")
def im_hist(img):   
    fig = plt.figure(num = 10, figsize=(8, 6))
    fig.clf() # Clear the figure
    ax = fig.gca()  # Get the current axes, creating one if necessary
    ax.hist(img.flatten(), bins = 256) # 256 bins for all different possible values
    plt.show()

im_hist(i_mono)

print("Plot a cumulative histogram")
def im_cdf(img):   
    fig = plt.figure(num = 11, figsize=(8, 6))
    fig.clf()
    ax = fig.gca()    
    ax.hist(img.flatten(), bins = 256, cumulative=True) # Try cumulative = False, it is same as previous figure
    plt.show()
    
im_cdf(i_mono)

# This returns a list with all figure numbers available
# See: https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
print(plt.get_fignums())

print("\n3. Equalize the image")
"""
In order just to simplify processing of that I might want to equalize it
that unevenness at the moment might indicate that there are some issues with
contrast in the in the video so I might want to equalize out the contrast and
and try and get a I'm working a simple set of numbers to work with
"""
"""
I have a more or less diagonal straight line for my CDF so I've certainly 
equalized out the values in the histogram and that may well make it easier for me 
to work with those values and try and extract some features
"""
from skimage import exposure

i_eq = exposure.equalize_hist(i_mono)
print("size of equalized image = ", i_eq.shape)
imshow(i_eq, cmap='gray')

im_hist(i_eq)
im_cdf(i_eq)

print("\n4. Denoising with Filters")

print("Add noise")
import skimage
i_n = skimage.util.random_noise(i_eq)
imshow(i_n, cmap="gray")
plt.show()

print("Use a Gaussian Filter")
"""
A Gaussian filter works by defining a patch of the image and determining 
the intensity value for the center pixel based on the weighted average of the pixels 
that surround it with coarser pixels having greater weight than more distant pixels 
so the average value is assigned to the center pixel and then the patch 
is moved and the process repeated until the entire image has been processed 

The Gaussian filters often produce a blurring effect because the average 
of the pixel intensity even in areas of the image where there are 
contrasting shades that define edges or corners

If I was trying to extract features from this if I'm looking at this as a computer
then actually that might be perfectly useful to me because I can see 
some really obvious features there are the eyes the nose the mouth and the t-shirt
under the jacket so I can you know see how that this may be useful to a computer 

But for me as a human looking at that that looks a bit blurred
"""
def gauss_filter(im, sigma = 10):    # patch/mask size is 10
    from scipy.ndimage.filters import gaussian_filter as gf
    import numpy as np
    return gf(im, sigma = sigma)   
i_g = gauss_filter(i_n)
imshow(i_g, cmap="gray")
plt.show()

print("Use a Median filter")
"""
A median filter works in the same way as a Gaussian filter except
that it applies the median value to the center pixel. This approach can be better
for removing small areas of noise in detailed images as it tends to keep the
pixel values that are in the same area of the image alike regardless of how
close they are to a contrasting area

Get my picture by now has removed some of the noise we could just compare that
to the original noisy image here so we can see there's a lot of kind of and
speckled bits of noise in there whereas here there's a bit less of that, it's
still a little blurred but it's certainly a cleaner image we can clearly
see what it is and we've cleaned it up and removed some of the noise
"""
def med_filter(im, size = 10):   # patch/mask size is 10
    from scipy.ndimage.filters import median_filter as mf
    import numpy as np
    return mf(im, size = size)     
i_m = med_filter(i_n)
imshow(i_m, cmap="gray")
plt.show()

print("\n5. Extract Features")

print("Sobel Edge Detection")
"""
EDGE DETECTORS USE GRADIENTS TO FIND CONTRAST AND PIXEL INTENSITY WHEN A MASK
IS MOVED HORIZONTALLY OR VERTICALLY 

We apply is a TWO-STAGE MASKING function that is applied to the pixel
intensity values in the image to calculate a gradient value BASED ON
CHANGES IN INTENSITY this mask is apply to find the horizontal gradient for each
pixel let's call it Gx and then this mask is applied to detect the vertical
gradients which we'll call Gy we then add the SQUARES of the x and y gradient
values that are calculated for each pixel and we take the SQUARE ROOT to
determine a gradient value for each pixel and we can calculate the INVERSE
TANGENT of those values to determine the angle of the edges that have been
detected
"""
def edge_sobel(image):
    from scipy import ndimage
    import skimage.color as sc
    import numpy as np
    image = sc.rgb2gray(image) # Convert color image to gray scale
    dx = ndimage.sobel(image, 1)  # horizontal derivative
    dy = ndimage.sobel(image, 0)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.amax(mag)  # normalize (Q&D)
    mag = mag.astype(np.uint8)
    return mag

i_edge = edge_sobel(i_m)          # input is median filtered image
imshow(i_edge, cmap="gray")

print("Harris Corner Detection")
"""
To detect corners in an image, you need to detect contrast in intensity 
in any direction

the harris corner detector algorithm works by testing patches of the image 
the patch is moved in multiple directions and the pixel intensity is compared 
for each position
- now in a featureless area of the image there's no significant difference so no
corner is detected 
- now let's try the patch in an area that contains an edge, in this case there is 
a difference in intensity when the patch is moved in one direction but 
not when moved along the edge so again there's no corner here 
- now let's position the patch in an area containing a corner a movement in any
direction results in a change of intensity so this looks like it might be
a corner
"""

def corner_harr(im, min_distance = 10):       # minimum distance
    from skimage.feature import corner_harris, corner_peaks
    mag = corner_harris(im)
    return corner_peaks(mag, min_distance = min_distance)

harris = corner_harr(i_eq, 10)    # INPUT IS EQUALIZED IMAGE

print("Harris corners =", harris)

def plot_harris(im, harris, markersize = 20, color = 'red'):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(6, 6))
    fig.clf()
    ax = fig.gca()    
    ax.imshow(np.array(im).astype(float), cmap="gray")
    ax.plot(harris[:, 1], harris[:, 0], 'r+', color = color, markersize=markersize)
    return 'Done'  

plot_harris(i_eq, harris)       # INPUT IS EQUALIZED IMAGE