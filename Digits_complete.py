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
from sklearn import metrics
from sklearn.datasets import load_digits
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
Nc_digits = 10


"""
Read the digits dataset"""
def test_digits() :
    digits = load_digits()
    X = digits.data
    y = digits.target
    X = preprocessing.scale(X)
#    X=X/X.max()
    return X   

"""
K means
"""
def kmeans(data):
    K_means = KMeans(init='random', n_clusters=Nc_digits, n_init=30)
    K_means.fit(data)
    centroids = K_means.cluster_centers_
    labels = K_means.labels_
    cluster1=[]
    cluster2=[]
    cluster3=[]
    cluster4=[]
    cluster5=[]
    cluster6=[]
    cluster7=[]
    cluster8=[]
    cluster9=[]
    cluster10=[]          
    for i in range (len(data)):
        if labels[i]==0:
            cluster1.append(data[i])
        elif labels[i]==1:
            cluster2.append(data[i])
        elif labels[i]==2:
            cluster3.append(data[i])
        elif labels[i]==3:
            cluster4.append(data[i])
        elif labels[i]==4:
            cluster5.append(data[i])
        elif labels[i]==5:
            cluster6.append(data[i])
        elif labels[i]==6:
            cluster7.append(data[i])
        elif labels[i]==7:
            cluster8.append(data[i])
        elif labels[i]==8:
            cluster9.append(data[i])
        elif labels[i]==9:
            cluster10.append(data[i])
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    cluster3 = np.asarray(cluster3)
    cluster4 = np.asarray(cluster4)
    cluster5 = np.asarray(cluster5)
    cluster6 = np.asarray(cluster6)
    cluster7 = np.asarray(cluster7)
    cluster8 = np.asarray(cluster8)
    cluster9 = np.asarray(cluster9)
    cluster10 = np.asarray(cluster10)                        
    return Fdigits(cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9,cluster10,centroids), labels
        
"""
Euclidean distance function
"""
def dist (Zp,Mij): 
    distance_one=0
    for j in range (len(Zp)):
        distance_one_temp = (Zp[j] - Mij[j])**2
        distance_one = distance_one + distance_one_temp
    distance_one = (distance_one)**0.5
    dist = distance_one
    return dist

"""
Fitness function for digits: Quantization Error 
"""
def Fdigits(cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9,cluster10,centroids):
    
    d1_temp = 0
    d2_temp = 0
    d3_temp = 0    
    d4_temp = 0
    d5_temp = 0
    d6_temp = 0 
    d7_temp = 0
    d8_temp = 0
    d9_temp = 0 
    d10_temp = 0
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
    for z in cluster4:
        d4 = dist(z,centroids[3])
        d4_temp+=d4
    d4_temp/= float(len(cluster4))
    for z in cluster2:
        d5 = dist(z,centroids[4])
        d5_temp+=d5
    d5_temp/= float(len(cluster5))
    for z in cluster6:
        d6 = dist(z,centroids[5])
        d6_temp+=d6
    d6_temp/= float(len(cluster6))
    for z in cluster7:
        d7 = dist(z,centroids[6])
        d7_temp+=d7
    d7_temp/= float(len(cluster7))
    for z in cluster8:
        d8 = dist(z,centroids[7])
        d8_temp+=d8
    d9_temp/= float(len(cluster8))
    for z in cluster9:
        d9 = dist(z,centroids[8])
        d9_temp+=d9
    d9_temp/= float(len(cluster9))
    for z in cluster10:
        d10 = dist(z,centroids[9])
        d10_temp+=d10
    d10_temp/= float(len(cluster10))
    return (d1_temp+d2_temp+d3_temp+d4_temp+d5_temp+d6_temp+d7_temp+d8_temp+d9_temp+d10_temp)/10


"""
Function takes in dataset and returns the three clusters formed for each particle
"""
def closest (data,particle,noofclusters):
    clus1 = []
    clus2 = []
    clus3 = []
    clus4 = []
    clus5 = []
    clus6 = []
    clus7 = []
    clus8 = []
    clus9 = []
    clus10 = []
    for i in range(len(data)):  
        d1 = dist(data[i],particle[0])
        d2 = dist(data[i],particle[1])
        d3 = dist(data[i],particle[2])
        d4 = dist(data[i],particle[3])
        d5 = dist(data[i],particle[4])
        d6 = dist(data[i],particle[5])
        d7 = dist(data[i],particle[6])
        d8 = dist(data[i],particle[7])
        d9 = dist(data[i],particle[8])
        d10 = dist(data[i],particle[9])
        d = min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)
        if (d == d1):
            clus1.append(data[i])
        elif (d == d2):
            clus2.append(data[i])
        elif (d == d3):
            clus3.append(data[i])
        elif (d == d4):
            clus4.append(data[i])
        elif (d == d5):
            clus5.append(data[i])
        elif (d == d6):
            clus6.append(data[i])
        elif (d == d7):
            clus7.append(data[i])
        elif (d == d8):
            clus8.append(data[i])
        elif (d == d9):
            clus9.append(data[i])
        elif (d == d10):
            clus10.append(data[i])
    return clus1, clus2, clus3,clus4,clus5,clus6,clus7,clus8,clus9,clus10
    
    
def label_pso (data,particle):
    
    label_pso_array = []

    for i in range(len(data)):  
        d1 = dist(data[i],particle[0])
        d2 = dist(data[i],particle[1])
        d3 = dist(data[i],particle[2])
        d4 = dist(data[i],particle[3])
        d5 = dist(data[i],particle[4])
        d6 = dist(data[i],particle[5])
        d7 = dist(data[i],particle[6])
        d8 = dist(data[i],particle[7])
        d9 = dist(data[i],particle[8])
        d10 = dist(data[i],particle[9])
        d = min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)
        if (d == d1):
            label_pso_array.append(0)
        elif (d == d2):
            label_pso_array.append(1)
        elif (d == d3):
            label_pso_array.append(2)
        elif (d == d4):
            label_pso_array.append(3)
        elif (d == d5):
            label_pso_array.append(4)
        elif (d == d6):
            label_pso_array.append(5)
        elif (d == d7):
            label_pso_array.append(6)
        elif (d == d8):
            label_pso_array.append(7)
        elif (d == d9):
            label_pso_array.append(8)
        elif (d == d10):
            label_pso_array.append(9)
    return label_pso_array
    
    
def labels (xdata,ydata, particle):
    label = []
    for i in range(len(xdata)):
        d1 = distance(xdata[i],ydata[i],particle[0])
        d2 = distance(xdata[i],ydata[i],particle[1])
        d3 = distance(xdata[i],ydata[i],particle[2])
        d4 = distance(xdata[i],ydata[i],particle[3])
        d5 = distance(xdata[i],ydata[i],particle[4])
        d6 = distance(xdata[i],ydata[i],particle[5])
        d7 = distance(xdata[i],ydata[i],particle[6])
        d8 = distance(xdata[i],ydata[i],particle[7])
        d9 = distance(xdata[i],ydata[i],particle[8])
        d10 = distance(xdata[i],ydata[i],particle[9])
        d = min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)
        if (d == d1):
            label.append(0)
        elif (d == d2):
            label.append(1)
        elif (d == d3):
            label.append(2)
        elif (d == d4):
            label.append(3)
        elif (d == d5):
            label.append(4)
        elif (d == d6):
            label.append(5)
        elif (d == d7):
            label.append(6)
        elif (d == d8):
            label.append(7)
        elif (d == d9):
            label.append(8)
        elif (d == d10):
            label.append(9)
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
Particle Class for digits datasets
"""
class Particle_digits:
    def __init__(self):

        self.derr = 0 
        self.dbest_pos = []
        self.dbest_err = -1 # this is set to -1 so we update after the first step
        self.dpos = []
        self.dvelocity = []       
        #initialize postion and velocity
        self.dpos = (self.init_position())
        self.dvelocity = (self.init_vel()) 
        self.dbest_pos = deepcopy(self.dpos)
        
    # function to initialize the particle positions
    def init_position (self):
        temp = np.zeros((Nc_digits,len(data_digits[0])))
        for i in range (Nc_digits):
            for j in range (len(data_digits[0])):
                temp[i][j] = 4*np.random.random_sample()-2
        return temp        
            
    # function to initialize the particle positions
    def init_vel (self):
        temp = np.zeros((Nc_digits,len(data_digits[0])))
        return temp
        
    #Evaluating performance of particle
    def evaluate (self):
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = closest(data_digits,self.dpos,Nc_digits)
        self.derr = Fdigits(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,self.dpos)
        if self.dbest_err == -1 or self.derr < self.dbest_err:
            self.dbest_err = deepcopy(self.derr)
            self.dbest_pos = deepcopy(self.dpos)
        return self.derr
    
    def update_vel (self,g_best_pos):         
        new_velocity_arr = []
        new_velocity = np.zeros((Nc_digits,len(data_digits[0])))
        for t in range (len(self.dvelocity)):
            for j in range (len(data_digits[0])):
                new_velocity[t][j] = w*self.dvelocity[t][j] + c1 * r1 * (self.dbest_pos[t][j] - self.dpos[t][j]) + c2 * r2 * (g_best_pos[t][j] - self.dpos[t][j])
        new_velocity_arr = new_velocity
        self.dvelocity = deepcopy(new_velocity_arr)
    
    def update_pos (self):  
        new_position_arr = []
        new_position = np.zeros((Nc_digits,len(data_digits[0])))
        for i in range (len(self.dpos)):
            for j in range (len(data_digits[0])):
                new_position[i][j] = self.dpos[i][j] + self.dvelocity[i][j]
        new_position_arr=(new_position)
        self.dpos = deepcopy(new_position_arr)   
        
        
"""
Particle Class for digits datasets PCA
"""
class Particle_digitsPCA:
    def __init__(self):

        self.derr = 0 
        self.dbest_pos = []
        self.dbest_err = -1 # this is set to -1 so we update after the first step
        self.dpos = []
        self.dvelocity = []       
        #initialize postion and velocity
        self.dpos = (self.init_position())
        self.dvelocity = (self.init_vel()) 
        self.dbest_pos = deepcopy(self.dpos)
        
    # function to initialize the particle positions
    def init_position (self):
        temp = np.zeros((Nc_digits,len(reduced_data[0])))
        for i in range (Nc_digits):
            for j in range (len(reduced_data[0])):
                temp[i][j] = 4*np.random.random_sample()-2
        return temp        
            
    # function to initialize the particle positions
    def init_vel (self):
        temp = np.zeros((Nc_digits,len(reduced_data[0])))
        return temp
        
    #Evaluating performance of particle
    def evaluate (self):
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = closest(reduced_data,self.dpos,Nc_digits)
        self.derr = Fdigits(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,self.dpos)
        if self.dbest_err == -1 or self.derr < self.dbest_err:
            self.dbest_err = deepcopy(self.derr)
            self.dbest_pos = deepcopy(self.dpos)
        return self.derr
    
    def update_vel (self,g_best_pos):         
        new_velocity_arr = []
        new_velocity = np.zeros((Nc_digits,len(reduced_data[0])))
        for t in range (len(self.dvelocity)):
            for j in range (len(reduced_data[0])):
                new_velocity[t][j] = w*self.dvelocity[t][j] + c1 * r1 * (self.dbest_pos[t][j] - self.dpos[t][j]) + c2 * r2 * (g_best_pos[t][j] - self.dpos[t][j])
        new_velocity_arr = new_velocity
        self.dvelocity = deepcopy(new_velocity_arr)
    
    def update_pos (self):  
        new_position_arr = []
        new_position = np.zeros((Nc_digits,len(reduced_data[0])))
        for i in range (len(self.dpos)):
            for j in range (len(reduced_data[0])):
                new_position[i][j] = self.dpos[i][j] + self.dvelocity[i][j]
        new_position_arr=(new_position)
        self.dpos = deepcopy(new_position_arr)
    
    
if __name__ == '__main__':
    
    while True:
        try:
            
    
            """Get Data"""  
            X_digits = test_digits()
            data_digits = deepcopy(X_digits)
    
            """PSO"""
    
            #Initialize the swarm 
            swarm_digits = []
            i=0
            j=0
            while i < (noofparticles):
                pdigit = Particle_digits()
                dcl1,dcl2,dcl3,dcl4,dcl5,dcl6,dcl7,dcl8,dcl9,dcl10 = closest(data_digits,pdigit.dpos,Nc_digits)
                if (len(dcl1) !=0) and (len(dcl2) !=0) and (len(dcl3) !=0) and (len(dcl4) !=0) and (len(dcl5) !=0) and (len(dcl6) !=0) and (len(dcl7) !=0) and (len(dcl8) !=0) and (len(dcl9) !=0) and (len(dcl10) !=0):
                    swarm_digits.append(pdigit)
                    i=i+1
                    print "All clusters in digits dataset are good"
                else:
                    i=i
                    print "Some clusters in digits dataset bad. Try again"
                    j+=1
    
            # Initialize the best position, velocity, error for wine and digits datasets
            best_pos_digits = []
            best_velocity_digits = []
            best_err_digits = -1
            i = 0  
            for i in range(tmax):
                j=0
                for j in range(len(swarm_digits)):
                    err_digits = swarm_digits[j].evaluate()
                    # If this particle is performing better than the rest Save its position velocity, and error
                    if err_digits < best_err_digits or best_err_digits == -1:
                        best_pos_digits = []
                        best_velocity_digits = []
                        best_pos_digits = deepcopy(swarm_digits[j].dpos)
                        best_velocity_digits = deepcopy(swarm_digits[j].dvelocity)
                        best_err_digits = err_digits
            
            
                    # Update the swarm based on the new positions
                j = 0
                for j in range (len(swarm_digits)):
                    swarm_digits[j].update_vel(best_pos_digits)
                    swarm_digits[j].update_pos()    
  
            print "Best Position: ", best_pos_digits
            print "---------PSO error is--------- ", best_err_digits    
            label_pso_array = np.asarray(label_pso(data_digits,best_pos_digits))
            print "---------PSO silhouette score is-------",silhouette_score(data_digits,label_pso_array)
    
            """K-Means"""
            Kmeans_error,labels_kmeans = kmeans(data_digits)
            silhouette_kmeans = silhouette_score(data_digits,labels_kmeans)
            print '----------K means error is-------------',Kmeans_error
            print '----------K means silhouette score pso is---------',silhouette_kmeans

            break
            
        except ZeroDivisionError:
            print 'Zero Division error. Trying again!!'
            
    while True:
        try:

###############################################################################
            # Visualize the K means results on PCA-reduced data
    
            reduced_data = PCA(n_components=2).fit_transform(data_digits)
            K_means_reduced = KMeans(init='random', n_clusters=Nc_digits, n_init=30)
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
            plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n''Centroids are marked with white cross')
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
            
            swarm_digits = []
            i=0
            j=0
            while i < (noofparticles):
                pdigit = Particle_digitsPCA()
                dcl1,dcl2,dcl3,dcl4,dcl5,dcl6,dcl7,dcl8,dcl9,dcl10 = closest(reduced_data,pdigit.dpos,Nc_digits)
                if (len(dcl1) !=0) and (len(dcl2) !=0) and (len(dcl3) !=0) and (len(dcl4) !=0) and (len(dcl5) !=0) and (len(dcl6) !=0) and (len(dcl7) !=0) and (len(dcl8) !=0) and (len(dcl9) !=0) and (len(dcl10) !=0):
                    swarm_digits.append(pdigit)
                    i=i+1
                    print "All clusters in digits dataset are good"
                else:
                    i=i
                    print "Some clusters in digits dataset bad. Try again"
                    j+=1
    
            # Initialize the best position, velocity, error for wine and digits datasets
            best_pos_digits = []
            best_velocity_digits = []
            best_err_digits = -1
            i = 0  
            for i in range(tmax):
                j=0
                for j in range(len(swarm_digits)):
                    err_digits = swarm_digits[j].evaluate()
                    # If this particle is performing better than the rest Save its position velocity, and error
                    if err_digits < best_err_digits or best_err_digits == -1:
                        best_pos_digits = []
                        best_velocity_digits = []
                        best_pos_digits = deepcopy(swarm_digits[j].dpos)
                        best_velocity_digits = deepcopy(swarm_digits[j].dvelocity)
                        best_err_digits = err_digits
            
            
                    # Update the swarm based on the new positions
                j = 0
                for j in range (len(swarm_digits)):
                    swarm_digits[j].update_vel(best_pos_digits)
                    swarm_digits[j].update_pos()    
  
            print "Best Position: ", best_pos_digits
            print "---------PSO error is--------- ", best_err_digits    
            
            Z = labels(xx.ravel(), yy.ravel(), best_pos_digits)
            Z = np.asarray(Z)
            Z = Z.reshape((xx.shape))
    
    
            plt.figure(2)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')

            plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            plt.scatter(best_pos_digits[:, 0], best_pos_digits[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
            plt.title('PSO clustering on the digits dataset (PCA-reduced data)\n''Centroids are marked with white cross')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.show()
            
            break            
            
            
        except ZeroDivisionError:
            print 'Faced divison by zero error. Trying Again.'
            
            