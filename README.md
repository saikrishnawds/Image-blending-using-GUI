# Image-blending-using-GUI
This repository contains my implementation of Facial Blending using Laplacian and Gaussian Pyramids by using GUI


Pyramid Blending
=================


(a) Write a function to implement Gaussian and Laplacian
pyramid,
𝑔𝑃𝑦𝑟, 𝑙𝑃𝑦𝑟 = 𝐶𝑜𝑚𝑝𝑢𝑡𝑒𝑃𝑦𝑟(𝑖𝑛𝑝𝑢𝑡_𝑖𝑚𝑎𝑔𝑒, 𝑛𝑢𝑚_𝑙𝑎𝑦𝑒𝑟𝑠),
Input arguments (for your reference): 𝑖𝑛𝑝𝑢𝑡_𝑖𝑚𝑎𝑔𝑒 is an input image
(grey, or RGB), 𝑛𝑢𝑚_𝑙𝑎𝑦𝑒𝑟𝑠 is the number of layers of the pyramid to
be computed. Depending on the size of 𝑖𝑛𝑝𝑢𝑡_𝑖𝑚𝑎𝑔𝑒,
𝑛𝑢𝑚_𝑙𝑎𝑦𝑒𝑟𝑠 needs to be checked if valid. If not, use the maximum value
allowed in terms of the size of 𝑖𝑛𝑝𝑢𝑡_𝑖𝑚𝑎𝑔𝑒.

Outputs: 𝑔𝑃𝑦𝑟, 𝑙𝑃𝑦𝑟 are the Gaussian pyramid and Laplacian pyramid
respectively.

1) The smoothing function: you can use built-in functions to generate the
Gaussian kernel (think about and research what the kernel size should
be, as well as [optional] ablation studies in your experiments), but you
need to use your own conv function or FFT function implemented in
Project 2. It’s a good time to improve your conv and/or FFT
implementation as you see fit.


2) The downsampler (for GaussianPyr) and upsampler (for
LaplacianPyr) use the simplest nearest neighbor interpolation.

(b) Write a simple GUI to create a black/white binary mask
image. The GUI can open an image (e.g. the foreground image that you
will use in blending); On the image, you can select a region of interest
using either a rectangle or an eclipse, [optional] even some free-form
region. Based on the opened image and the selected regions, the GUI can
generate a black/white mask image of the same size as the opened image,
in which the selected region(s) are white and the remaining black.
Note: For this GUI, you can search online and reuse whatever functions
you find useful and can be put together as a single self-contained
module to realize the aforementioned mask generation functionality.
But, you need to finish this on your own.


(c) On top of the functions in (a) and (b), write a function to
implement Laplacian pyramid blending (see instructions in page 23 of
lecture note 17)

