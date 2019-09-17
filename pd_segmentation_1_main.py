import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_float
from skimage.color import rgb2gray, gray2rgb
from skimage.segmentation import mark_boundaries
import time
import matplotlib.image as mpimg
exec(open('/Users/Salim_Andre/Desktop/IMA/PRAT/code/pd_segmentation_1.py').read())

### DATASET

PATH_img = '/Users/Salim_Andre/Desktop/IMA/PRAT/' # path to my own images

swans=mpimg.imread(PATH_img+'swans.jpg');
baby=mpimg.imread(PATH_img+'baby.jpg'); 
	
img_set = [data.astronaut(), data.camera(), data.coins(), data.checkerboard(), data.chelsea(), \
	data.coffee(), data.clock(), data.hubble_deep_field(), data.horse(), data.immunohistochemistry(), \
	data.moon(), data.page(), data.rocket(), swans, baby]
	
### IMAGE

I=img_as_float(img_set[4]);

###	PARAMETERS FOR 1-HOMOLOGY GROUPS

n_superpixels=400;
RV_epsilon=180;
gauss_sigma=0.5;
n_events=6;
#n_pxl_min_ = 10;
density_excl=0.0;
plot_pd=False; 

### RESULTS FOR 1-HOMOLOGY GROUPS
	
dim=1;

start_exc = time.time()
img_sym, segments_pxl = pd_segmentation_1(n_superpixels, I, RV_epsilon, gauss_sigma, n_events, density_excl, plot_pd);
end_exc = time.time()

### SHOW RESULTS FOR 1-HOMOLOGY GROUPS

# 1-holes in image
plt.rcParams["figure.figsize"] = [7,7]
#listplots=[321,322,323,324,325,326];
for i, seg in enumerate(segments_pxl):
	#plt.subplot(listplots[i]);
	plt.axis('off')
	plt.imshow(rgb2gray(I), cmap='gray', interpolation='nearest')
	plt.scatter(seg[:,1],seg[:,0], c='r')#, ms=5)
	start=seg[0,:];
	end=seg[-1,:];
	end_loop=np.array([end,start]);
	plt.plot(seg[:,1],seg[:,0], '-r')#, ms=5)
	plt.plot(end_loop[:,1],end_loop[:,0], '-r')#, ms=5)
	#plt.savefig('/Users/Salim_Andre/Desktop/IMA/PRAT/projet/figures_experiments/cascades/coffee_'+str(i+1)+'.png')
	plt.show()
	
'''
# 1-holes in image
plt.rcParams["figure.figsize"] = [7,7]
plt.imshow(rgb2gray(I), cmap='gray', interpolation='nearest')
#plt.imshow(np.abs(ndimage.sobel(rgb2gray(I))))
#plt.imshow(np.abs(gaussian_filter(ndimage.sobel(rgb2gray(I)), sigma=5.)))
#listplots=[321,322,323,324,325,326];
for i, seg in enumerate(segments_pxl):
	#plt.subplot(listplots[i]);
	plt.axis('off')
	plt.scatter(seg[:,1],seg[:,0], c='r', alpha=0.01)#, ms=5)
	start=seg[0,:];
	end=seg[-1,:];
	end_loop=np.array([end,start]);
	plt.plot(seg[:,1],seg[:,0], '-r')#, ms=5)
	plt.plot(end_loop[:,1],end_loop[:,0], '-r')#, ms=5)
	#plt.savefig('/Users/Salim_Andre/Desktop/IMA/PRAT/projet/figures_experiments/cascades/coffee_'+str(i+1)+'.png')
plt.show()
'''
print('\nExecution time: {:2.2f} seconds \n'.format(end_exc - start_exc))
