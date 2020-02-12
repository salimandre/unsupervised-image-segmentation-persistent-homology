import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_float
from skimage.color import rgb2gray, gray2rgb
from skimage.segmentation import mark_boundaries
import time
import matplotlib.image as mpimg
exec(open('/Users/Salim_Andre/Desktop/IMA/PRAT/code/pd_segmentation_0.py').read())
exec(open('/Users/Salim_Andre/Desktop/IMA/PRAT/code/tree.py').read())

### DATASET

PATH_img = '/Users/Salim_Andre/Desktop/IMA/PRAT/' # path to my own images

swans=mpimg.imread(PATH_img+'swans.jpg');
baby=mpimg.imread(PATH_img+'baby.jpg'); 
	
img_set = [data.astronaut(), data.camera(), data.coins(), data.checkerboard(), data.chelsea(), \
	data.coffee(), data.clock(), data.hubble_deep_field(), data.horse(), data.immunohistochemistry(), \
	data.moon(), data.page(), data.rocket(), swans, baby]
	
### IMAGE

I=img_as_float(img_set[0]);

###	PARAMETERS FOR 0-HOMOLOGY GROUPS

mode='customized';
n_superpixels=10000;
RV_epsilon=30;
gauss_sigma=0.5;
list_events=[800];
n_pxl_min_ = 30;
density_excl=0.0;
entropy_thresh_=1.1;
plot_pd=False;

'''
mode = 'standard'
n_superpixels=10000;
'''
### RESULTS FOR 0-HOMOLOGY GROUPS

start = time.time()
if mode=='standard':
	img_sym, segments_pd, img_labels = pd_segmentation_0(mode, n_superpixels, I);
else:
	img_sym, segments_pd, img_labels = pd_segmentation_0(mode, n_superpixels, I, RV_epsilon, gauss_sigma, list_events, n_pxl_min_, entropy_thresh_, density_excl, plot_pd);
end = time.time()

### PLOTS

#stock colors

my_colors = [(0,0,0)]*len(segments_pd);
for i, seg in enumerate(segments_pd):
	my_colors[i]=tuple(np.random.rand(1,3)[0]);

# SEGMENTATION

# image
plt.rcParams["figure.figsize"] = [10,10]
plt.imshow(I)
plt.axis('off')
plt.show()

# image segmentation with pd
plt.rcParams["figure.figsize"] = [10,10]
plt.imshow(np.ones(img_sym.shape), cmap='gray', vmin=0, vmax=1)
for i, seg in enumerate(segments_pd):
		plt.plot(seg[:,1], seg[:,0], 's', ms=5.1, color=my_colors[i])
plt.axis('off')
#plt.title('PCD\'s most persistent 0-homology groups')
plt.show()

# image segmentation with pd
plt.rcParams["figure.figsize"] = [10,10]
plt.imshow(rgb2gray(img_sym), cmap='gray', interpolation='nearest')
for i, seg in enumerate(segments_pd):
		plt.plot(seg[:,1], seg[:,0], 's', ms=5.1, color=my_colors[i])
plt.axis('off')
#plt.title('Pixels removed are near boundaries and complex regions')
plt.show()

# SAMPLING

Isampling=np.zeros((I.shape[0],I.shape[1],3));
for l in range(1,np.max(img_labels)+1):
	Isampling[img_labels==l]=my_colors[l-1];
plt.imshow(Isampling)
plt.axis('off')
#plt.title('Image segmention after sampling')
plt.show()
'''
plt.imshow(mark_boundaries(I, img_labels))
for i, seg in enumerate(segments_pd):
		plt.plot(seg[:,1], seg[:,0], 's', ms=5.1, color=my_colors[i])
plt.axis('off')
plt.title('Image segmention after sampling')
plt.show()
'''
# IMAGE SEGMENTATION

plt.rcParams["figure.figsize"] = [10,10]
plt.axis('off')
plt.imshow(mark_boundaries(I, img_labels))
#plt.title('Image segmentation')
plt.show()

# IMAGE RECOVERING

Iregions=np.zeros(I.shape);
for l in range(1,np.max(img_labels)+1):
	Iregions[img_labels==l]=np.mean(I[img_labels==l],axis=0);

if len(I.shape)==2:
	plt.imshow(Iregions, cmap='gray', interpolation='nearest');
else:
	plt.imshow(Iregions);
#plt.title('Image recovering from segments')
plt.axis('off')
plt.show()

#print(np.sum(img_labels==0))
	
print('\nExecution time: {:2.2f} seconds \n'.format(end - start))

print('\nNumber of segments: {:} \n'.format(np.max(img_labels)))

err_1 = np.sum(np.abs((Iregions-I)*255.))/np.prod(Iregions.shape);
print('Mean error norm 1 per pixel: {:2.2f}'.format(err_1));

err_2 = np.sqrt(np.sum(((Iregions-I)*255.)**2)/np.prod(Iregions.shape));
print('Mean error norm 2 per pixel: {:2.2f}'.format(err_2));

'''
collage photos
349 * 344
'''
