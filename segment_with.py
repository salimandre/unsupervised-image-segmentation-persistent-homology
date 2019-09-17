
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage import exposure
from skimage.util import img_as_float
from skimage.color import rgb2gray, gray2rgb

def segment_with(I, seg_method): 
 
 # PARAMETERS
	
	# Felzenswalb
	scale_=400; 
	sigma_=.5; 
	min_size_=500;

	# SLIC
	n_segments_=100; #15
	compactness_=10;
	sigma_=1

	# Quickshift
	kernel_size_=20
	max_dist_=45
	ratio_=0.5

	# Watershed
	markers_=10
	compactness_=0.001

# SEGMENTATION METHODS

	if seg_method=='Otsu Thresholding':
		
		I=rgb2gray(I);
		thd=threshold_otsu(I);
		Ib=I<thd;
		plt.imshow(Ib, cmap='gray', interpolation='nearest')
		plt.axis('off')
		plt.show()

		hist, bins_center = exposure.histogram(I)
		plt.plot(bins_center, hist, lw=2)
		plt.axvline(thd, color='r', ls='--')
		plt.show()
		return Ib

	# FELZENSWALB'S METHOD
	if seg_method=='Felzenswalb':
		
		plt.axis('off')
		plt.title('Felzenswald\'s method')
		segments_fz = felzenszwalb(I, scale_, sigma_, min_size_);
		plt.imshow(mark_boundaries(I, segments_fz))
		plt.show()
		J=I;
		for l in range(np.max(segments_fz)+1):
			J[segments_fz==l]=np.mean(I[segments_fz==l], axis=0);
		if len(I.shape)==2:
			plt.imshow(J, cmap='gray', interpolation='nearest');
		else:
			plt.imshow(J);
		plt.axis('off')
		plt.show()
		return segments_fz

	# SLIC'S METHOD
	if seg_method=='SLIC':
		
		plt.axis('off')
		plt.title('SLIC\'s method')
		segments_slic = slic(I, n_segments=n_segments_, compactness=compactness_, sigma=sigma_)
		plt.imshow(mark_boundaries(I, segments_slic))
		plt.show()
		J=I;
		for l in range(np.max(segments_slic)+1):
			J[segments_slic==l]=np.mean(I[segments_slic==l], axis=0);
		if len(I.shape)==2:
			plt.imshow(J, cmap='gray', interpolation='nearest');
		else:
			plt.imshow(J);
		plt.axis('off')
		plt.show()
		return segments_slic

	# QUICKSHIFT'S METHOD
	if seg_method=='Quickshift':
		
		plt.axis('off')
		plt.title('Quickshift\'s method')
		segments_quick =quickshift(I, kernel_size=kernel_size_, max_dist=max_dist_, ratio=ratio_)
		plt.imshow(mark_boundaries(I, segments_quick))
		plt.show()
		return segments_quick
		
	#  WATERSHED'S METHOD
	if seg_method=='Watershed':
		
		plt.axis('off')
		plt.title('Watershed\'s method')
		gradient = sobel(rgb2gray(I));
		segments_watershed = watershed(gradient, markers=markers_, compactness=compactness_);
		plt.imshow(mark_boundaries(I, segments_watershed));
		plt.show()

		return segments_watershed
