import numpy as np
import matplotlib.pyplot as plt
import sys
# python gudhi library path
sys.path.append('/Users/Salim_Andre/Desktop/GUDHI_2.3.0/build/cython');
import gudhi as gd
from scipy.ndimage import gaussian_filter
import networkx as nx
from sklearn.neighbors.kde import KernelDensity
from random import choice
from scipy import ndimage
import collections

def pd_segmentation_1(n_superpixels_,img_,RV_epsilon_, gauss_sigma_, n_events_, density_excl_=0.05, plot_pd_=False):
	# required: Networkx as nx, Gudhi as gd, from sklearn.neighbors.kde import KernelDensity
	# RV_epsilon_ : maximum edge length of the Rips-Vietoris complex
	# sigma_ : 0 < param <= 1 coeff for gaussian blur the bigger the more blur
	# dim_ : order of homology, also dimension of the R-V complex
	# density_param_ : percent of points to be exclude using a gaussian kernel density filter
	
	dim_ = 1;
		
	height, width = img_.shape[:2];
	if len(img_.shape) >2:
		n_col_chan = 3; 
	else:
		n_col_chan=1;
		   
	step = int(round(0.5*(np.sqrt(height * width / n_superpixels_)-1))); # make sure to get n_segments superpixels 
	# extend img by symmetry to have a perfect cover with square patches
	dh=(2*step+1)-(height%(2*step+1));
	dw=(2*step+1)-(width%(2*step+1));
	dhI=img_[-dh:,:];
	img_=np.concatenate((img_,dhI[::-1,:]),axis=0);
	dwI=img_[:,-dw:];
	img_=np.concatenate((img_,dwI[:,::-1]),axis=1);  
	# subsampling                                    
	grid_y, grid_x = np.mgrid[:img_.shape[0], :img_.shape[1]];                                           
	means_y = grid_y[step::2*step+1, step::2*step+1];
	means_x = grid_x[step::2*step+1, step::2*step+1];

	if gauss_sigma_ > 0:
		# gaussian blur
		Iblur=gaussian_filter(img_, sigma=gauss_sigma_*step/4); 											  
	else:
		Iblur=img_;
		del img_;
	# from image to cloud point data
	pcd = np.dstack((means_y,means_x,Iblur[means_y, means_x]*255)).reshape((-1,n_col_chan+2));
		
	nb_points = pcd.shape[0]; # real number of data points
	print('\nNumber of initial superpixels: {:}'.format(nb_points)) # real number of data points after density filtering
	
	#ndimage.sobel(rgb2gray(I))
	
	if density_excl_ > 0:
		# apply density filtering
		kde = KernelDensity(kernel='gaussian', bandwidth=20).fit(pcd);
		pcd_density=kde.score_samples(pcd);
		ranked_density= sorted(pcd_density, reverse=True);
		n_excl=int(nb_points*density_excl_);
		thresh_density= ranked_density[-n_excl:][0];
		# filter point cloud data with density threshold
		excl_pcd=pcd[pcd_density<=thresh_density,:2];
		pcd=pcd[pcd_density>thresh_density,:];
		# update number of data points
		nb_points = pcd.shape[0]; # updated real number of data points
	else:
		excl_pcd=np.zeros((0,2));
		
	print('\nNumber of superpixels after density filtering: {:}'.format(nb_points)) # real number of data points after density filtering
	
	# from PCD to Rips-Vietoris complex
	Rips_complex_sample = gd.RipsComplex(points = pcd,max_edge_length=RV_epsilon_);
	Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2);
	# compute persistence diagram on simplex tree structure
	diag_Rips = Rips_simplex_tree_sample.persistence(); # (dim, (birth_date, death_date))
	
	if plot_pd_==True:
		# compute persistence diagram for dimension dim_
		plt=gd.plot_persistence_diagram(diag_Rips,legend=True);
		plt.show()
		
	betti_nb_0 = Rips_simplex_tree_sample.betti_numbers()[0];
	print('\nBetti number beta_0 at oo: {:}'.format(betti_nb_0));
	betti_nb_1 = Rips_simplex_tree_sample.betti_numbers()[1];
	print('\nBetti number beta_1 at oo: {:}'.format(betti_nb_1));

	# build covering tree with minimum value
	G=nx.Graph()
	ppairs_0=[pair for pair in Rips_simplex_tree_sample.persistence_pairs() if len(pair[0])==1];
	list_edges_0=[tuple(pair[1]) for pair in ppairs_0 if pair[1]!=[]];
	G.add_nodes_from(range(nb_points));
	G.add_edges_from(list_edges_0);

	ppairs_1=[pair for pair in Rips_simplex_tree_sample.persistence_pairs() if len(pair[0])==2];
	list_key_edges_1=[tuple(pair[0]) for pair in ppairs_1];
	death_by_interv=[a[1][1] for a in Rips_simplex_tree_sample.persistence() if a[0]==1]
	ind_sorted=np.argsort(np.argsort(death_by_interv));
	key_edges = [pair for pair in [list_key_edges_1[ind] for ind in ind_sorted[:n_events_] ]];
	G_cycles=[];
	# add edge which creates a new class of cycle
	# search cycles with dft in O(|V|)
	for edge in key_edges:
		G.add_edges_from([edge]);
		G_cycles=G_cycles+nx.cycle_basis(G) #nx.find_cycle(G)];
		G.remove_edges_from([edge]);
		
	segments_pxl=[pcd[nodes,:2] for nodes in G_cycles];
	
	return Iblur, segments_pxl
