from PIL import Image
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward
import pylab as plb
import matplotlib.patches as patches


class Image_Analysis(object):
    
    
    def __init__(self, image, plot = False):
        self.a = []
        self.p = []
        self.v = []
        self.image = image
        self.plot = plot
    
    
    def analyse_smear(self, n_clusters):
        self.n_clusters = n_clusters
        if self.n_clusters < 3:
            print 'Requires at least 3 clusters'
            
        # Generate data
        A = Image.open(self.image)
        B = A.convert('L')
        C = np.array(B)
        
        self.conv_image = C
        print type(self.conv_image)
        #self.conv_image = np.array()
        # Downsample the image by a factor of 4
        self.conv_image = self.conv_image[::2, ::2] + self.conv_image[1::2, ::2] + self.conv_image[::2, 1::2] + self.conv_image[1::2, 1::2]
        X = np.reshape(self.conv_image, (-1, 1))
        
        # Define the structure A of the data. Pixels connected to their neighbors.
        connectivity = grid_to_graph(*self.conv_image.shape)
        
        # Compute clustering
        print("Compute structured hierarchical clustering...")
        ward = Ward(n_clusters=self.n_clusters, connectivity=connectivity).fit(X)
        self.label = np.reshape(ward.labels_, self.conv_image.shape)
        print("Number of pixels: ", self.label.size)
        print("Number of clusters: ", np.unique(self.label).size)
        
        for l in range(self.n_clusters):
            self.a.append(plb.contour(self.label == l, contours=1, colors=[plb.cm.spectral(l / float(self.n_clusters)), ]))
            self.p.append(self.a[l].collections[0].get_paths()[0])
            self.v.append(self.p[l].vertices)
        del self.v[0]    
        
    
    def plot_paths(self):
        fig = plb.figure()
        ax = fig.add_subplot(111)
        patch = []
        
        for i in range(len(self.p)):
            patch.append(patches.PathPatch(self.p[i], facecolor='orange', lw=2))
        for i in range(len(patch)):
            ax.add_patch(patch[i])    
        
        ax.set_xlim(-2,100)
        ax.set_ylim(-2,100)
        plb.show()
        
        # Plot the results on an image
        plb.figure(figsize=(5, 5))
        plb.imshow(self.conv_image, cmap=plb.cm.gray)
        for l in range(self.n_clusters):
            plb.contour(self.label == l, contours=1, colors=[plb.cm.spectral(l / float(self.n_clusters)), ])
        #pl.contour(label == 14, contours=1, colors=[pl.cm.spectral(14 / float(self.n_clusters)), ])
        plb.xticks(())
        plb.yticks(())
        plb.show()
        
        
    def analyse_image(self):
        self.analyse_smear(3)
        if self.plot:
            self.plot_paths()
        return self.v
       
if __name__ == '__main__':
    
    Culley = Image_Analysis(image = 'tests/test5.png', plot = True)
    
    Culley.analyse_image()