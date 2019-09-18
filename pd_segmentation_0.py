import numpy as np
import matplotlib.pyplot as plt
import sys
# python gudhi library path
sys.path.append('/Users/Salim_Andre/Desktop/GUDHI_2.3.0/build/cython');
import gudhi as gd
from scipy.ndimage import gaussian_filter
import networkx as nx
from sklearn.neighbors.kde import KernelDensity
import time

def pd_segmentation_0(mode_, n_superpixels_,img_, RV_epsilon_=30, gauss_sigma_=0.5, list_events_=[350], n_pxl_min_=10, entropy_thresh_=0.05, density_excl_=0.05, plot_pd_=False):
	# required: Networkx as nx, Gudhi as gd, from sklearn.neighbors.kde import KernelDensity
	# n_superpixels_: wanted number of superpixels, algo will take closest N_suppxl s.t. N_suppxl * square_size = N_pixels
	# RV_epsilon_ : maximum edge length of the Rips-Vietoris complex
	# sigma_ : 0 < param <= 1 coeff for gaussian blur the bigger the more blur
	# dim_ : order of homology, also dimension of the R-V complex (=0)
	# density_excl_ : percent of points to be exclude using a gaussian kernel density filter
	# n_pxl_min_ : minimum number of pixels per segments
	# list_events_ : sequence of cuts in the covering tree of minimum value e.g. [n1, n2, n3] will produce n1 cuts, then n2 - n1 cuts, then n3 - n2 cuts. where n1<n2<n3 !  
	# plot_pd_: bool, plots the persistence diagram for 0 homology groups if True
		
	height, width = img_.shape[:2];
	height_0=height;
	width_0=width;
	if len(img_.shape) >2:
		n_col_chan = 3; 
	else:
		n_col_chan=1;
		 
	n_pixels = np.prod(img_.shape[:2]);
	list_squares=[4, 9, 16, 25, 36, 49, 64]
	list_pos=[(1,0), (1,1), (2,1), (2,2), (3,2), (3,3), (4,3)]; 
	list_steps=[2, 3, 4, 5, 6, 7, 8];

	if mode_=='standard':
		gauss_sigma_ = 0.5;
		n_pxl_min_= 15;
		entropy_thresh_= .15;
		density_excl_= 0.;
		plot_pd_=False;
		list_squares=[4, 9, 16, 25, 36, 49, 64]#[25, 36, 49, 64];
		list_pos=[(1,0), (1,1), (2,1), (2,2), (3,2), (3,3), (4,3)]#[(2,2), (3,2), (3,3), (4,3)];
		list_steps=[2, 3, 4, 5, 6, 7, 8]#[5, 6, 7, 8];
		
	i_step = np.argmin([np.abs(n_pixels/s - n_superpixels_) for s in list_squares]);
	step = list_steps[i_step];
	
	step_up_j=list_pos[i_step][0]
	step_down_j=list_pos[i_step][1]
	
	step_up_i=list_pos[i_step][1]
	step_down_i=list_pos[i_step][0]
	
	dh = int(np.ceil(height/step))*step - height;
	dw = int(np.ceil(width/step))*step - width;

	if dh>0:
		dhI=img_[-dh:,:];
		img_=np.concatenate((img_, dhI[::-1,:]), axis=0);
	if dw>0:
		dwI=img_[:,-dw:];
		img_=np.concatenate((img_, dwI[:,::-1]), axis=1); 
	
	grid_y, grid_x = np.mgrid[:img_.shape[0], :img_.shape[1]];                                           
	means_y = grid_y[list_pos[i_step][0]::step, list_pos[i_step][1]::step];
	means_x = grid_x[list_pos[i_step][0]::step, list_pos[i_step][1]::step];
	
	if gauss_sigma_ > 0:
		# gaussian blur
		Iblur=gaussian_filter(img_, sigma=gauss_sigma_*np.floor(0.5*step)/4); 											  
	else:
		Iblur=img_;
		del img_;
	# from image to cloud point data
	pcd = np.dstack((means_y,means_x,Iblur[means_y, means_x]*255)).reshape((-1,n_col_chan+2));
	
	nb_points = pcd.shape[0]; # real number of data points
	
	if density_excl_ > 0:
		print('\nNumber of initial superpixels: {:}'.format(nb_points)) # real number of data points after density filtering
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
		print('\nNumber of superpixels after density filtering: {:}'.format(nb_points)) # real number of data points after density filtering
	else:
		excl_pcd=np.zeros((0,2));

	if mode_=='standard': # compute RV_epsilon from ratio
		ratio=np.prod(height_0*width_0)/nb_points;
		print('ratio = ', ratio)
		RV_epsilon_ = np.ceil(0.5*ratio+10);
	
	# print input variables
	print('\nInput variables: :')
	print('size of superpixels : ', step,' * ', step)
	print('Number of superpixels: {:}'.format(nb_points))
	print('RV epsilon = ', RV_epsilon_)
	print('Blur sigma = ', gauss_sigma_*np.floor(0.5*step)/4)
	print('% of removed pixels by density filtering = ', density_excl_)
	print('min size of segments = ', n_pxl_min_*step*step, 'pixels ')
	print('max % of removed pixels per cut = ', entropy_thresh_)
	print('plot persistence diagram: ',plot_pd_);
	
	# from PCD to Rips-Vietoris complex
	Rips_complex_sample = gd.RipsComplex(points = pcd,max_edge_length=RV_epsilon_);
	Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=1);

	# compute persistence diagram on simplex tree structure
	diag_Rips = Rips_simplex_tree_sample.persistence(); # (dim, (birth_date, death_date))
	
	if plot_pd_==True:
		# compute persistence diagram for dimension dim_
		diag_Rips_0=Rips_simplex_tree_sample.persistence_intervals_in_dimension(0);
		print('lamost plot')
		plt=gd.plot_persistence_diagram([(0, interval) for interval in diag_Rips_0], max_plots=0, alpha=0.1,legend=True)
		plt.show()
	
	# stock persistent pairs -> key topological events
	ppairs=Rips_simplex_tree_sample.persistence_pairs();

	betti_0 = Rips_simplex_tree_sample.betti_numbers()[0];
	print('\nBetti number beta_0 at oo: {:}\n'.format(betti_0))
	
	# stock persistent pairs -> key topological events
	key_edges_0=[tuple(pair[1]) for pair in Rips_simplex_tree_sample.persistence_pairs() if len(pair[0])==1][:-betti_0];

	# build covering tree with minimum value using 0-1 persistence pairs
	G=nx.Graph()
	G.add_nodes_from(range(nb_points));
	G.add_edges_from(key_edges_0);

	if mode_=='standard':
		
		if n_pxl_min_*step*step<=200 and ratio<10:
			list_events_=[500, 1000];
		if n_pxl_min_*step*step>200 and ratio<10:
			list_events_=[200, 500, 1000];
		
		if n_pxl_min_*step*step<=100 and ratio>=10 and ratio<18:
			list_events_=[600];	
		if n_pxl_min_*step*step>100 and n_pxl_min_*step*step<=200 and ratio>=10 and ratio<18:
			list_events_=[200, 600];
		if n_pxl_min_*step*step>200 and ratio>=10 and ratio<18:
			list_events_=[125, 300, 700];
			
		if n_pxl_min_*step*step<=100 and ratio>18:
			list_events_=[500];		
		if n_pxl_min_*step*step>=100 and n_pxl_min_*step*step<200 and ratio>18:
			list_events_=[200, 500];
		if n_pxl_min_*step*step>=200 and n_pxl_min_*step*step<300 and ratio>18:
			list_events_=[150, 500];
		if n_pxl_min_*step*step>=300 and ratio>18:
			list_events_=[75, 275, 500];
			
		c0=(list_events_[0]-betti_0)*(betti_0<list_events_[0]) + 1*(betti_0>=list_events_[0]);
		list_events_[0]=c0;

	print('list of cuts: ',list_events_, '\n')
	list_events=[None]+[-nc for nc in list_events_]
	cuts=[]
	for i in range(len(list_events_)):
		cuts = cuts + [[tuple(np.sort(edge)) for edge in key_edges_0[list_events[i+1]:list_events[i]]] ];
			 
	tree = Tree(G);
	
	n_expand=len(list_events_);
	for i in range(n_expand):
		tree.expand(cuts[i], size = n_pxl_min_, proba = entropy_thresh_);
	
	#print(tree.as_str())

	#print('depth = ', tree.get_depth(),'\n')
	
	#print('number of nodes = ', tree.count,'\n')

	in_segments = [leaf.pixels for leaf in tree.get_leaves()];
	
	n_kept_pixels = sum([len(leaf_pxl) for leaf_pxl in in_segments]);
	
	n_leaves = len(in_segments);
	#print('nb leaves = ', n_leaves,'\n')
	
	#print('PCD = ', n_kept_pixels, ' pixels | ', round(100.*n_kept_pixels/len(tree.pixels),2), ' %\n')
	
	n_removed_pixels = len(tree.pixels)-n_kept_pixels;
	print('loss = ', n_removed_pixels, ' pixels | ', round(100.*n_removed_pixels/len(tree.pixels),2), ' %\n')

	out_segments = [0]*n_removed_pixels;
	i=0;
	for cut in tree.out_pixels: #start sampling for least persistent removed pixels !
		for seg in cut:
			for pxl in seg:
				out_segments[i]=pxl;
				i+=1;
	
	# image	with labels
	img_labels=np.zeros(Iblur.shape[:2],dtype='int64');
	height, width = Iblur.shape[:2];
		
	# segments with pixel positions
	pcd_in_segments = [pcd[seg,:2] for seg in in_segments];
	pcd_out_segments = pcd[out_segments,:2];
	
	# label all superpixels which have not been removed by kernel density filter
	for l, segment_l in enumerate(pcd_in_segments):
		for pxl in segment_l:
			i=int(pxl[0]);
			j=int(pxl[1]);
			img_labels[i-step_down_i:i+step_up_i+1,j-step_down_j:j+step_up_j+1]=l+1;
			
	# label all superpixels which have been removed by kernel density filter by with uniform distribution on neigbhors
		
	percent_out = 100.*len(out_segments)/pcd.shape[0];
	#print('sampling clusters for {:} removed pixels, {:2.2f} % of all pixels\n'.format(len(out_segments),percent_out));
	distrib=[np.power(np.sum(img_labels==l), -0.12) for l in range(1, np.max(img_labels)+1)];
	#step_i=[step_down_i,step_up_i];
	#step_j=[step_down_j,step_up_j];
	for pxl in pcd_out_segments[::-1,:]:
		i=int(pxl[0]);
		j=int(pxl[1]);
		label_V_ij=[]
		k=1;
		while len(label_V_ij)<1:
			#V_k=[(i*2*step_i[i>0], j*2*step_j[j>0]) for i in range(-k,k+1) for j in range(-k,k+1) if abs(i)==k or abs(j)==k];
			for delta in [(-2*k*step_down_i,-2*k*step_down_j), (-2*k*step_down_i,0), (-2*k*step_down_i,+2*k*step_up_j), (0,-2*k*step_down_j), (0,+2*k*step_up_j), (+2*k*step_up_i,-2*k*step_down_j), (+2*k*step_up_i,0), (+2*k*step_up_i,+2*k*step_up_j)]:	
				if 0<=i+delta[0] and i+delta[0]<height and j+delta[1]<width and 0<=j+delta[1] and img_labels[i+delta[0], j+delta[1]]!=0.:
					label_V_ij = label_V_ij+[img_labels[i+delta[0], j+delta[1]]];
				k=k+1;	
		if len(set(label_V_ij))>1:
			distrib_V_ij=[0]*len(label_V_ij);
			for ind, l in enumerate(label_V_ij):
				distrib_V_ij[ind]=distrib[l-1];
			distrib_V_ij=distrib_V_ij/sum(distrib_V_ij);
			
			img_labels[i-step_down_i:i+step_up_i+1,j-step_down_j:j+step_up_j+1]=np.random.choice(label_V_ij, p=distrib_V_ij); # sample label based on neigbhors and label distribution
			
		else:
			
			img_labels[i-step_down_i:i+step_up_i+1,j-step_down_j:j+step_up_j+1]=label_V_ij[0]; # closest neighbor
		label_V_ij=[];
		
	# remove symmetric expansion 
	img_labels=img_labels[:height_0,:width_0];
	
	return Iblur, pcd_in_segments, img_labels











