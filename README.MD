# Unsupervised image segmentation using persistent homology theory

In the early of 20s century Algebraic topology provided, thanks to Poincaré, a general framework to classify shapes. Indeed the **Euler characteristic** equal to the alternating sum of the Betti numbers is a **topological invariant**. Roughly these numbers count the number of distinct objects in the domain, the number of holes and the number of voids they contain etc...

<p align="center">
  <img src="img/homology_classification.jpg" width="35%">
</p>

**Topological Data Analysis** (TDA) is the field which apply these theorical tools in order to proceed data analysis. But these latter characteristics cannot be used straight forward because of the uncertainty of the datas and because of the sensitivity of Betti numbers to minor outliers in the data set. Therefore to tackle this issue the main tool TDA is using is **persistent homology**, in which the invariants are in the form of **persistence diagrams** also called **barcodes**. Topological invariants will then quantify the stability of geometric features with respect to degradation such as noise or artefacts

<p align="center"> 
<img src="img/filtration_barcodes_chazal_funda.jpg" width="50%">
</p>
<p align="center"> 
Credits @ Frédéric Chazal, Bertrand Michel
</p>

We used TDA framework to perform unsupervised image segmentation. The set of images provided by the skimage library from python and the python library Gudhi @INRIA to produce simplicial complexes and persistence diagrams. 

The main **procedure** was as followed : 

+ Apply a small gaussian blur to the image to remove isolated pixels (outliers)
+ Take a random sample of points from the image called **superpixels**. We obtain a 3D cloud point if image is gray a 5D cloud point if the image is in color. 
+ Compute the **Nested Rips-Vietoris complexes** from those points 
  + Computed nested RV complexes for radius ε from 0 to infinity 
  + Set a value for edges : the distance between the two vertices. Value of vertices is set to 0. 
  + Set a value for each simplex by taking the max value of all its edges (Method called age filter) 
+ Compute the persistent pairs for homology groups for dimensions 0 and 1 and for 1 and 2. For dimensions 0 and 1 these are pairs (c,e) where c is a non zero element of a group H0(Kε) for some ε and e isanedgesuchthattheaddofetoKε′ forε′ ≥εmakescvanish,i.e.cisazeroforH0(Kε′). 
+ With the set of all edges for persistent pairs of dimensions 0-1 we can compute the covering tree over our cloud data point with minimum value. 
+ Then we can remove the nc −1 edges which vanish the nc −1 most persistent component connexes of our filtration. This gives us the nc most persistent connexe components. 
+ Or we can add edges which gave birth to the most persistent cycles through the filtration. Then find loops with a traversal algorithm. 
+ The most persistent connexe components and the most persistent cycles give each a segmentation of our images.
