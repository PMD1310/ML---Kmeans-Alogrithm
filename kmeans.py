import math  # For pow and sqrt
import sys
from random import shuffle, uniform
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


###_Pre-Processing_###
def LoadData(fileName):
    # Read the file, splitting by lines
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()

    items = []

    for i in range(1, len(lines)):
        line = lines[i].split(',')
        itemFeatures = []

        for j in range(len(line) - 1):
            v = float(line[j])  # Convert feature value to float
            itemFeatures.append(v)  # Add feature value to dict

        items.append(itemFeatures)

    shuffle(items)

    return items


###_Auxiliary Function_###
def MinMaxOfColumn(items):
    n = len(items[0])
    minimum = [sys.maxsize for i in range(n)]
    maximum = [-sys.maxsize - 1 for i in range(n)]

    for item in items:
        for f in range(len(item)):
            if (item[f] < minimum[f]):
                minimum[f] = item[f]

            if (item[f] > maximum[f]):
                maximum[f] = item[f]

    return minimum, maximum


def DistancebyEuclid(x, y):
    SquaredRootDistance = 0  # The sum of the squared differences of the elements
    for i in range(len(x)):
        SquaredRootDistance += math.pow(x[i] - y[i], 2)

    return math.sqrt(SquaredRootDistance)  # The square root of the sum


def OriginalMeans(items, k, clusterMin, clusterMax):
    # Initialize means to random numbers between
    # the min and max of each column/feature

    f = len(items[0])  # number of features
    means = [[0 for i in range(f)] for j in range(k)]

    for mean in means:
        for i in range(len(mean)):
            # Set value to a random float
            # (adding +-1 to avoid a wide placement of a mean)
            mean[i] = uniform(clusterMin[i] + 1, clusterMax[i] - 1)

    return means


def MeansUpdating(n, mean, item):
    for i in range(len(mean)):
        m = mean[i]
        m = (m * (n - 1) + item[i]) / float(n)
        mean[i] = round(m, 3)
    # print ("Updated Means \n",mean)
    return mean


def ClusterFinding(means, items):
    clusters = [[] for i in range(len(means))]  # Init clusters

    for item in items:
        # Classifying item into a cluster
        index = Classifying(means, item)

        # Add item to cluster
        clusters[index].append(item)
    return clusters


###_Core Functions_###
def Classifying(means, item):
    # Classifying item to the mean with minimum distance

    minimum = sys.maxsize
    index = -1

    for i in range(len(means)):
        # Find distance from item to mean
        dis = DistancebyEuclid(item, means[i])

        if (dis < minimum):
            minimum = dis
            index = i
    print("\n Data being assigned to different clusters:", index)
    return index


def CalculateMeans(k, items, maxIterations=100000):
    # Find the minimum and maximum for columns
    clusterMin, clusterMax = MinMaxOfColumn(items)

    # Initialize means at random points
    means = OriginalMeans(items, k, clusterMin, clusterMax)

    # Initialize clusters, the array to hold
    # the number of items in a class
    clusterSizes = [0 for i in range(len(means))]

    # An array to hold the cluster an item is in
    belongsTo = [0 for i in range(len(items))]

    # Calculate means
    for e in range(maxIterations):
        # If no change of cluster occurs, halt
        noChange = True
        for i in range(len(items)):
            item = items[i]
            # Classifying item into a cluster and update the
            # corresponding means.

            index = Classifying(means, item)

            clusterSizes[index] += 1
            means[index] = MeansUpdating(clusterSizes[index], means[index], item)

            # Item changed cluster
            if (index != belongsTo[i]):
                noChange = False

            belongsTo[i] = index

        # Nothing changed, return
        if (noChange):
            break

    return means


###_Main_###
def main():
    items = LoadData('data.txt')

    k = 3

    means = CalculateMeans(k, items)
    clusters = ClusterFinding(means, items)
    print("\n Means of the Dataset:\n", means)
    print("\n Clustering sets are: \n", clusters)

    # To determine how many clusters are efficient, we can use Elbow Curve.
    # Loading the dataset again to get the best cluster value.
    dataset = pd.read_csv('Iris.csv')
    x = dataset.iloc[:, [1, 2, 3, 4]].values

    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()


if __name__ == "__main__":
    main()