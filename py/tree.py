class Tree(object):
    def __init__(self, graph=None):
        self.graph = graph  
        self.parent = None
        self.children = None
        if not not self.graph:
            self.pixels = graph.nodes;
        else:
            self.pixels = None
        self.depth = 0
        self.state = 0;
        self.key = 0;
        self.count = 1;
        self.out_pixels = None;  
          
    def get_root(self, count='no'):
        if self.depth==0:
            return self
        else:
            node = self;
            if count!='yes':	
                for i in range(self.depth):
                    node=node.parent;
            else:	
                for i in range(self.depth):
                    node=node.parent;
                    node.count=node.count + 1;
            return node;
            
    def get_leaves(self):
        if not self.children:
            yield self
	
        else:
            for child in self.children:
                for leaf in child.get_leaves():
                    yield leaf

    def get_depth(self):
        depth=0;
        for leaf in self.get_leaves():
            if leaf.depth>depth:
                depth=leaf.depth;
        return depth

    def expand(self,list_edges_to_be_removed, size=10, proba=0.05):
        out_pixels=[]
        H=self.graph;
        for leaf in self.get_leaves():
            
            if leaf.state==0:
                
                subH=nx.Graph(H.subgraph(leaf.pixels));
                
                if set(subH.edges()) & set(list_edges_to_be_removed):
                       
                    subH.remove_edges_from(list_edges_to_be_removed)
                    cc = [list(a) for a in nx.connected_components( subH ) ];
                    
                    if len(cc)>1:
                        
                        if size>0:

                            nodes_toberemoved=[-1]*len(cc);
                            loss = 0;
                            n_nodes_tbr=0
                            max_size_c=size;
                            for j, c in enumerate(cc):
                                size_c = len(c);
                                if size_c <= size: #small cc
                                    nodes_toberemoved[n_nodes_tbr]=j
                                    loss+=size_c;
                                    n_nodes_tbr+=1;
                                if size_c > max_size_c: #big cc
                                    max_size_c = size_c;
                            nodes_toberemoved=nodes_toberemoved[:n_nodes_tbr];
                            cpt_not_small = len(cc)-len(nodes_toberemoved);
                            bool_expand = loss/len(leaf.pixels) <= proba or self.get_depth()==0;
                            bool_pause = (max_size_c/len(leaf.pixels) >= 1.-proba and not self.get_depth()==0 and max_size_c/len(self.get_root().pixels)<0.3);
                        else:
                            nodes_toberemoved=[];
                            bool_expand=True;
                            bool_pause=False;
                            cpt_not_small=1000;
                            
                        if bool_expand and cpt_not_small>0 and not bool_pause: #expand
                            if len(nodes_toberemoved)>0:
                                out_pixels=out_pixels+[cc[i] for i in nodes_toberemoved];
                            cc = [cc[i] for i in range(len(cc)) if i not in nodes_toberemoved];
                            leaf.children = [Tree() for c in cc];
                            for i, child in enumerate(leaf.children):
                                child.pixels=cc[i];
                                child.parent=leaf;
                                child.depth = leaf.depth + 1;
                                child.key = child.get_root(count='yes').count-1;
                        elif bool_pause:
                            leaf.state=0; # continue                          
                            
                        else:
                            leaf.state=1; #block
        if not self.get_root().out_pixels:
            self.get_root().out_pixels=[out_pixels];
        else:
            self.get_root().out_pixels+=[out_pixels];
                                              
    def as_str(self, level=0):
        ret = "\t"*level+repr(len(self.pixels))+"\n"
        if not not self.children:
            for child in self.children:
                ret += child.as_str(level+1)
        return ret
                    
                    
'''                 
H= nx.Graph();
H.add_path([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

tree = Tree(H)

re_1=[(3,4),(5,6)];

tree.expand(re_1)

print(tree.as_str())

re_2=[(2,3),(3,4),(5,6),(7,8),(9,10)];

tree.expand(re_2)

print(tree.as_str())
'''
