# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:24:10 2015

@author: paresh
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import distance_metrics


#Values used for PSO            
w = 0.72
c1 = 1.49
c2 = 1.49
r1 = np.random.uniform()
r2 = np.random.uniform()
tmax = 1000
noofparticles=10
Nc_wine = 3



""" 
Read the wine data
"""
def get_data (path):
    data=np.genfromtxt(path, delimiter=",")
    I = np.arange(len(data))
    np.random.shuffle(I)
    data = data[I]
    X = data[:,1:14]
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    return X
    


"""
K means
"""
def kmeans(data):
    K_means = KMeans(init='random', n_clusters=Nc_wine, n_init=30)
    K_means.fit(data)
    centroids = K_means.cluster_centers_
    labels = K_means.labels_
    cluster1=[]
    cluster2=[]
    cluster3=[]

    for i in range (len(data)):
        if labels[i]==0:
            cluster1.append(data[i])
        elif labels[i]==1:
            cluster2.append(data[i])
        elif labels[i]==2:
            cluster3.append(data[i])
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    cluster3 = np.asarray(cluster3)
    return F(cluster1,cluster2,cluster3,centroids),labels,centroids
    

   
"""
Euclidean distance function
"""
def dist (Zp,Mij): #Zp = pth data vector ; Mij = Centroid vector for Cluster j for particle i
    distance_one=0
    for j in range (len(Zp)):
        distance_one_temp = (Zp[j] - Mij[j])**2
        distance_one = distance_one + distance_one_temp
    distance_one = (distance_one)**0.5
    dist = distance_one
    return dist

"""
Fitness function: Quantization Error
"""
def F(cluster1,cluster2,cluster3,centroids):
    
    d1_temp = 0
    d2_temp = 0
    d3_temp = 0    
    for z in cluster1:
        d1 = dist(z,centroids[0])
        d1_temp+=d1
    d1_temp/= float(len(cluster1))
        
    for z in cluster2:
        d2 = dist(z,centroids[1])
        d2_temp+=d2
    d2_temp/= float(len(cluster2))

    for z in cluster3:
        d3 = dist(z,centroids[2])
        d3_temp+=d3
    d3_temp/= float(len(cluster3))   
    return (d1_temp+d2_temp+d3_temp)/3


"""
Function takes in dataset and returns the three clusters formed for each particle
"""

def closest (data,particle):
    
    clus1 = []
    clus2 = []
    clus3 = []

    for i in range(len(data)):  
        d1 = dist(data[i],particle[0])
        d2 = dist(data[i],particle[1])
        d3 = dist(data[i],particle[2])
        d = min(d1,d2,d3)
        if (d == d1):
            clus1.append(data[i])
        elif (d == d2):
            clus2.append(data[i])
        elif (d == d3):
            clus3.append(data[i])
    return clus1, clus2, clus3
    

def label_pso (data,particle):
    
    label_pso_array = []

    for i in range(len(data)):  
        d1 = dist(data[i],particle[0])
        d2 = dist(data[i],particle[1])
        d3 = dist(data[i],particle[2])
        d = min(d1,d2,d3)
        if (d == d1):
            label_pso_array.append(0)
        elif (d == d2):
            label_pso_array.append(1)
        elif (d == d3):
            label_pso_array.append(2)
    return label_pso_array
    
    
def labels (xdata,ydata, particle):
    label = []
    for i in range(len(xdata)):
        d1 = distance(xdata[i],ydata[i],particle[0])
        d2 = distance(xdata[i],ydata[i],particle[1])
        d3 = distance(xdata[i],ydata[i],particle[2])
        d = min(d1,d2,d3)
        if (d == d1):
            label.append(0)
        elif (d == d2):
            label.append(1)
        elif (d == d3):
            label.append(2)
    return label
        
def distance (dx, dy, C):
    distance_one=0
    distance_one_temp = (C[0] - dx)**2 + (C[1] - dy)**2
    distance_one = (distance_one_temp)**0.5
    dist = distance_one
    return dist           

"""
Silhouette Score functions
"""
def silhouette_score(X, labels,metric='euclidean', sample_size=None,random_state=None):
    if sample_size is not None:
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            raise ValueError('Distance matrix cannot be precomputed')
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, metric=metric))
    
def silhouette_samples(X, labels, metric='euclidean'):
    metric = distance_metrics()[metric]
    n = labels.shape[0]
    A = np.array([intra_cluster_distance(X, labels, metric, i)
                  for i in range(n)])
    B = np.array([nearest_cluster_distance(X, labels, metric, i)
                  for i in range(n)])
    sil_samples = (B - A) / np.maximum(A, B)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)

def intra_cluster_distance(X, labels, metric, i):
    indices = np.where(labels == labels[i])[0]
    if len(indices) == 0:
        return 0.
    a = np.mean([metric(X[i], X[j]) for j in indices if not i == j])
    return a

def nearest_cluster_distance(X, labels, metric, i):
    label = labels[i]
    b = np.min(
            [np.mean(
                [metric(X[i], X[j]) for j in np.where(labels == cur_label)[0]]
            ) for cur_label in set(labels) if not cur_label == label])
    return b


"""
Particle Class
"""
class Particle:
    def __init__(self):

        self.err = 0 
        self.best_pos = []
        self.best_err = -1 # this is set to -1 so we update after the first step
        self.pos = []
        self.velocity = []
        
        #initialize postion and velocity
        self.pos = (self.init_position())
        self.velocity = (self.init_vel()) 
        self.best_pos = deepcopy(self.pos)
        
    # function to initialize the particle positions
    def init_position (self):
        temp = np.zeros((Nc_wine,len(data_pso[0])))
        for i in range (Nc_wine):
            for j in range (len(data_pso[0])):
                temp[i][j] = np.random.random_sample()
        return temp   

            
    # function to initialize the particle positions
    def init_vel (self):
        temp = np.zeros((Nc_wine,len(data_pso[0])))
        return temp
        
        
    #Evaluating performance of particle
    def evaluate (self):
        cluster1,cluster2,cluster3 = closest(data_pso,self.pos)
        self.err = F(cluster1,cluster2,cluster3,self.pos)
        if self.best_err == -1 or self.err < self.best_err:
            self.best_err = deepcopy(self.err)
            self.best_pos = deepcopy(self.pos)
        return self.err
    
    def update_vel (self,g_best_pos):         
        new_velocity_arr = []
        new_velocity = np.zeros((Nc_wine,len(data_pso[0])))
        for t in range (len(self.velocity)):
            for j in range (len(data_pso[0])):
                new_velocity[t][j] = w*self.velocity[t][j] + c1 * r1 * (self.best_pos[t][j] - self.pos[t][j]) + c2 * r2 * (g_best_pos[t][j] - self.pos[t][j])
        new_velocity_arr = new_velocity
        self.velocity = deepcopy(new_velocity_arr)
    
    def update_pos (self):
        t2 = len(self.pos)     
        new_position_arr = []
        new_position = np.zeros((Nc_wine,len(data_pso[0])))
        for i in range (t2):
            for j in range (len(data_pso[0])):
                new_position[i][j] = self.pos[i][j] + self.velocity[i][j]
        new_position_arr=(new_position)
        self.pos = deepcopy(new_position_arr)
    

"""
Particle Class
"""
class ParticlePCA:
    def __init__(self):

        self.err = 0 
        self.best_pos = []
        self.best_err = -1 # this is set to -1 so we update after the first step
        self.pos = []
        self.velocity = []
        
        #initialize postion and velocity
        self.pos = (self.init_position())
        self.velocity = (self.init_vel()) 
        self.best_pos = deepcopy(self.pos)
        
    # function to initialize the particle positions
    def init_position (self):
        temp = np.zeros((Nc_wine,len(reduced_data[0])))
        for i in range (Nc_wine):
            for j in range (len(reduced_data[0])):
                temp[i][j] = np.random.random_sample()
        return temp    
            
    # function to initialize the particle positions
    def init_vel (self):
        temp = np.zeros((Nc_wine,len(reduced_data[0])))
        return temp

        
    #Evaluating performance of particle
    def evaluate (self):
        cluster1,cluster2,cluster3 = closest(reduced_data,self.pos)
        self.err = F(cluster1,cluster2,cluster3,self.pos)
        if self.best_err == -1 or self.err < self.best_err:
            self.best_err = deepcopy(self.err)
            self.best_pos = deepcopy(self.pos)
        return self.err
    
    def update_vel (self,g_best_pos):         
        new_velocity_arr = []
        new_velocity = np.zeros((Nc_wine,len(reduced_data[0])))
        for t in range (len(self.velocity)):
            for j in range (len(reduced_data[0])):
                new_velocity[t][j] = w*self.velocity[t][j] + c1 * r1 * (self.best_pos[t][j] - self.pos[t][j]) + c2 * r2 * (g_best_pos[t][j] - self.pos[t][j])
        new_velocity_arr = new_velocity
        self.velocity = deepcopy(new_velocity_arr)
    
    def update_pos (self):
        t2 = len(self.pos)     
        new_position_arr = []
        new_position = np.zeros((Nc_wine,len(reduced_data[0])))
        for i in range (t2):
            for j in range (len(reduced_data[0])):
                new_position[i][j] = self.pos[i][j] + self.velocity[i][j]
        new_position_arr=(new_position)
        self.pos = deepcopy(new_position_arr)

################################     MAIN     ###################################    
    
if __name__ == '__main__':

    while True:
        try:
    
            """Get Data"""  
            X = get_data('wine.data')
            data_pso = deepcopy(X)
    
            """PSO"""
    
            #Initialize the swarm
            swarm = []    
            i=0
            while i < (noofparticles):
                p = Particle()
                cl1, cl2,cl3 = closest(data_pso,p.pos)
                if (len(cl1) !=0) and (len(cl2) !=0) and (len(cl3) !=0):
                    swarm.append(p)
                    i=i+1
#                    print "All clusters good"
                else:
                    i=i
#                    print "Some cluster bad. Try again"

    
            # Initialize the best position, velocity, error
            best_pos = []
            best_velocity = []
            best_err = -1
            i = 0   
    
            for i in range(tmax):
            # Iterate the swarm and evaluate their position on the function
                j = 0
                for j in range(len(swarm)):
                    err = swarm[j].evaluate()
                    # If this particle is performing better than the rest Save its position velocity, and error
                    if err < best_err or best_err == -1:
                        best_pos = []
                        best_velocity = []
                        best_pos = deepcopy(swarm[j].pos)
                        best_velocity = deepcopy(swarm[j].velocity)
                        best_err = err

                # Update the swarm based on the new positions
                j = 0
                for j in range (len(swarm)):
                    swarm[j].update_vel(best_pos)
                    swarm[j].update_pos()
    
#            print "Best Position PSO: ", best_pos
            print "---------PSO error is--------- ", best_err    
            label_pso_array = np.asarray(label_pso(data_pso,best_pos))
            print "---------PSO silhouette score is-------",silhouette_score(data_pso,label_pso_array)
    
    
            """K-Means"""
            Kmeans_error,labels_kmeans,centroids_kmeans = kmeans(data_pso)
            silhouette_kmeans = silhouette_score(data_pso,labels_kmeans)
            print '----------K means error is-------------',Kmeans_error
            print '----------K means silhouette score is---------',silhouette_kmeans


            
            """PSO Hybrid"""
            
            #Initialize the swarm
            swarm = []    
            i=0
            while i < (noofparticles-1):
                p = Particle()
                cl1, cl2,cl3 = closest(data_pso,p.pos)
                if (len(cl1) !=0) and (len(cl2) !=0) and (len(cl3) !=0):
                    swarm.append(p)
                    i=i+1
#                    print "All clusters good"
                else:
                    i=i
#                    print "Some cluster bad. Try again"

    
            p = Particle()
            p.pos = deepcopy(centroids_kmeans)
            swarm.append(p)
    
            # Initialize the best position, velocity, error
            best_pos = []
            best_velocity = []
            best_err = -1
            i = 0   
    
            for i in range(tmax):
            # Iterate the swarm and evaluate their position on the function
                j = 0
                for j in range(len(swarm)):
                    err = swarm[j].evaluate()
                    # If this particle is performing better than the rest Save its position velocity, and error
                    if err < best_err or best_err == -1:
                        best_pos = []
                        best_velocity = []
                        best_pos = deepcopy(swarm[j].pos)
                        best_velocity = deepcopy(swarm[j].velocity)
                        best_err = err

                # Update the swarm based on the new positions
                j = 0
                for j in range (len(swarm)):
                    swarm[j].update_vel(best_pos)
                    swarm[j].update_pos()
    
#            print "Best Position PSO-Hybrid: ", best_pos
            print "---------PSO-Hybrid error is--------- ", best_err    
            label_pso_array = np.asarray(label_pso(data_pso,best_pos))
            print "---------PSO-Hybrid silhouette score is-------",silhouette_score(data_pso,label_pso_array)


    
            break
        except ZeroDivisionError:
            print 'Faced divison by zero error. Trying Again.'
    

    while True:
        try:
            
###############################################################################
            # Visualize the K means results on PCA-reduced data
    
            reduced_data = PCA(n_components=2).fit_transform(data_pso)
            K_means_reduced = KMeans(init='random', n_clusters=Nc_wine, n_init=30)
            K_means_reduced.fit(reduced_data)
            h = .02
            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = K_means_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.figure(1)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')
            plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

            # Plot the centroids as a white X
            centroids = K_means_reduced.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
            plt.title('K-means clustering on the wine dataset (PCA-reduced data)\n''Centroids are marked with white cross')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.show()
#    
####################################################################################
            ## Visualize the PSO results for PCA-reduced data
            #    
            ## Initialize the swarm
            swarm = []    
            i=0
            while i < (noofparticles):
                p = ParticlePCA()
                cl1, cl2,cl3 = closest(reduced_data,p.pos)
                if (len(cl1) !=0) and (len(cl2) !=0) and (len(cl3) !=0):
                    swarm.append(p)
                    i=i+1
#                    print "All clusters good"
                else:
                    i=i
#                    print "Some cluster bad. Try again"

            # Initialize the best position, velocity, error
            best_pos = []
            best_velocity = []
            best_err = -1
            i = 0   
            for i in range(tmax):
                # Iterate the swarm and evaluate their position on the function
                j = 0
                for j in range(len(swarm)):
                    err = swarm[j].evaluate()
                    # If this particle is performing better than the rest Save its position velocity, and error
                    if err < best_err or best_err == -1:
                        best_pos = []
                        best_velocity = []
                        best_pos = deepcopy(swarm[j].pos)
                        best_velocity = deepcopy(swarm[j].velocity)
                        best_err = err
            
                # Update the swarm based on the new positions
                j = 0
                for j in range (len(swarm)):
                    swarm[j].update_vel(best_pos)
                    swarm[j].update_pos()
    
#            print "Best Error PSO - PCA reduced: ", best_err
#            print "Best Position: ", best_pos
    
            Z = labels(xx.ravel(), yy.ravel(), best_pos)
            Z = np.asarray(Z)
            Z = Z.reshape((xx.shape))
    
    
            plt.figure(2)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')

            plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            plt.scatter(best_pos[:, 0], best_pos[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
            plt.title('PSO clustering on the wine dataset (PCA-reduced data)\n''Centroids are marked with white cross')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.show()
            
#    
####################################################################################
            ## Visualize the Hybrid PSO results for PCA-reduced data
            #    
            ## Initialize the swarm
            swarm = []    
            i=0
            while i < (noofparticles - 1):
                p = ParticlePCA()
                cl1, cl2,cl3 = closest(reduced_data,p.pos)
                if (len(cl1) !=0) and (len(cl2) !=0) and (len(cl3) !=0):
                    swarm.append(p)
                    i=i+1
#                    print "All clusters good"
                else:
                    i=i
#                    print "Some cluster bad. Try again"
                    
            p = ParticlePCA()
            p.pos = deepcopy(centroids)
            swarm.append(p)

            # Initialize the best position, velocity, error
            best_pos = []
            best_velocity = []
            best_err = -1
            i = 0   
            for i in range(tmax):
                # Iterate the swarm and evaluate their position on the function
                j = 0
                for j in range(len(swarm)):
                    err = swarm[j].evaluate()
                    # If this particle is performing better than the rest Save its position velocity, and error
                    if err < best_err or best_err == -1:
                        best_pos = []
                        best_velocity = []
                        best_pos = deepcopy(swarm[j].pos)
                        best_velocity = deepcopy(swarm[j].velocity)
                        best_err = err
            
                # Update the swarm based on the new positions
                j = 0
                for j in range (len(swarm)):
                    swarm[j].update_vel(best_pos)
                    swarm[j].update_pos()
    
#            print "Best Error PSO - PCA reduced: ", best_err
#            print "Best Position: ", best_pos
    
            Z = labels(xx.ravel(), yy.ravel(), best_pos)
            Z = np.asarray(Z)
            Z = Z.reshape((xx.shape))
    
    
            plt.figure(3)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')

            plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            plt.scatter(best_pos[:, 0], best_pos[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
            plt.title('PSO clustering on the wine dataset (PCA-reduced data)\n''Centroids are marked with white cross')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.show()
            
            break            
            
            
        except ZeroDivisionError:
            print 'Faced divison by zero error. Trying Again.'