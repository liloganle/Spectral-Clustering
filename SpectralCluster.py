# -*- coding:utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

def distance(vector1, vector2):
    return np.sum(np.power(vector1 - vector2, 2), axis=1)

def initialize(data_set, K=2):
    """initialize the centroids"""
    (num_samples, dim) = data_set.shape
    centroids = np.zeros((K, dim))
    for i in range(K):
        indexs = np.random.randint(i*num_samples/K, (i+1)*num_samples/K)
        centroids[i,:] = data_set[indexs,:]
    
    return centroids

def kmeans(data_set, K=2):
    """The K-means algorithm"""
    (num_samples, dim) = data_set.shape
    distance_array = np.zeros((num_samples, K))
    
    data_label = np.ones(num_samples)
    centroids = initialize(data_set, K)
    
    while True:
        for i in range(K):
            distance_array[:,i] = distance(data_set,centroids[i,:])
            
        temp_label = np.argmin(distance_array, axis=1)
        for i in range(K):
            centroids[i,:] = np.mean(data_set[temp_label == i], axis=0)

        if(all(data_label == temp_label)):
            break
        else:
            data_label = temp_label
    print "Congratulations, cluster completed!"
    return centroids, data_label

def draw(data_set, data_label, K = 2):
    """To draw the result of Cluster algorithm"""
    dim = data_set.shape[1]
    data_marks = ["ro","bo","ko","go","mo","r^","b^","k^","g^","m^"]
    #centroid_marks = "yD"

    if(dim != 2):
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1
    
    #draw all the samples and centroids
    for i in range(K):
        temp = data_set[data_label == i]
        plt.plot(temp[:,0], temp[:,1], data_marks[i])
    plt.title("The Result of Spectral Clustering")
    plt.show()

def gaussian(vector1, vector2, sigma = 1):
    return np.exp(-np.sum(np.power(vector1 - vector2, 2),axis = 1)/(2*sigma**2))

def ng_algorithm(data_set, K, sigma = 1):
    """Spectral Clustering:Ng algorithm
    data_set: training data
    K:the number of clusters"""
    (num_samples, dim) = data_set.shape
    similarity_array = np.zeros((num_samples, num_samples))
    degree_array = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        distances = distance(data_set, data_set[i,:])
        indexs = np.argsort(distances)
        indexs = indexs[1:K+1]
        similarity_array[i,indexs] = gaussian(data_set[indexs,:], data_set[i,:], sigma)
        #degree_array[i,i] = np.sum(similarity_array[i,:])

    similarity_array = (similarity_array.T + similarity_array)/2.0
    for i in range(num_samples):
        degree_array[i,i] = np.sum(similarity_array[i,:])
    laplace_array = degree_array - similarity_array
    degree_inverse = np.linalg.inv(np.power(degree_array, 0.5))
    laplace_sym = np.dot(degree_inverse.dot(laplace_array), degree_inverse)

    (eigenvalue, eigenvector) = np.linalg.eig(laplace_sym)
    indexs = np.argsort(eigenvalue[eigenvalue > 0])
    indexs = indexs[:K]
    evect_array = eigenvector[:,indexs]
    for i in range(len(evect_array)):
        evect_array[i,:] = evect_array[i,:]/np.sqrt(np.sum(evect_array[i,:]**2))
    
    (centroids, labels) = kmeans(evect_array)

    return centroids, labels

if __name__=="__main__":
    data = np.loadtxt("./data.txt")
    num_samples = data.shape[0]
    
    data_labels = np.concatenate((np.zeros(num_samples/2), np.ones(num_samples/2)))
    K = 4 #range(2,21) #the number of cluster
    #accuracy_array = np.zeros(len(K))
    sigma = 1 #np.arange(0.1,10,0.1)
    #accuracy_array = np.zeros(len(sigma))

    plt.plot(data[:,0],data[:,1],'o')
    plt.title("Original Draw")
    plt.show()
    plt.plot(data[:100,0],data[:100,1],"ro",data[100:,0],data[100:,1],"bo")
    plt.title("The Correct Result")
    plt.show()


    (centroids, labels) = ng_algorithm(data, K, sigma)
    draw(data, labels)
    #print "The centroids of cluster are:\n",centroids
    print "The Accuracy of Spectral Clustering is:",np.sum(labels == data_labels)/float(num_samples)



