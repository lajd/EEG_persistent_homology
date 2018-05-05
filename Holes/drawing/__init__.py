
'''
##################################################################################
# Visualization functions for the statistical properties of the cycle generators #
##################################################################################
'''

__author__ = """\n""".join(['Giovanni Petri (petri.giovanni@gmail.com)']);

import numpy as np

def cycle_persistence_distribution(Gen_dict,W=None,tag=' ',nbins=100):
	import networkx as nx;
	import matplotlib.pyplot as plt
	persistence=[];
	if W==None:
		W=0;
		for cycle in Gen_dict:
			if float(cycle.end)>float(W):
				W=cycle.end;
	for cycle in Gen_dict:
		persistence.append(float(cycle.persistence_interval())/float(W));
	plt.figure();
	# the histogram of the data
	n, bins, patches = plt.hist(persistence, nbins, normed=True, facecolor='green', alpha=0.75)
	if tag!=' ':
		plt.savefig(tag+'_persistence_distribution.png');
	return n, bins

def cycle_length_distribution(Gen_dict,tag=' ',nbins=100):
	import networkx as nx;
	import matplotlib.pyplot as plt
	length_cycles=[];

	for cycle in Gen_dict:
		length_cycles.append(len(cycle.composition));
	plt.figure();
	n, bins, patches = plt.hist(length_cycles, nbins, normed=True, facecolor='green', alpha=0.75)
	if tag!=' ':
		plt.savefig(tag+'_cycle_length_distribution.png');
	return n, bins

def cycle_start_distribution(Gen_dict,W=None,tag=' ',nbins=100):
	import networkx as nx;
	import matplotlib.pyplot as plt
	start_cycles=[];
	if W==None:
		W=0;
		for cycle in Gen_dict:
			if float(cycle.end)>float(W):
				W=float(cycle.end);
	for cycle in Gen_dict:
		start_cycles.append(float(cycle.start)/float(W));
	plt.figure();
	n, bins, patches = plt.hist(start_cycles, nbins, normed=True, facecolor='green', alpha=0.75)
	if tag!=' ':
		plt.savefig(tag+'_cycle_start_distribution.png');
	return n, bins

def barcode_creator(cycles,W=None,sizea=10,sizeb=10,verbose=False, title=''):
    import matplotlib.pyplot as plt;
    import numpy as np
    if W==None:
		W=0;
		for cycle in cycles:
			if float(cycle.end)>float(W):
				W=float(cycle.end);
    if verbose==True:
	    print('Maximum W=',W)
    fig=plt.figure(figsize=(sizea,sizeb));
    L=len(cycles);
    factor=np.sqrt(L);
    for i,cycle in enumerate(cycles):
	    plt.plot([float(cycle.start)/float(W),float(cycle.end)/float(W)],[factor*(L-i), factor*(L-i)],'o-');
    plt.xlabel('epsilon')
    plt.ylabel('arbitrary ordering of homology generators')
    plt.title(title)


