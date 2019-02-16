import kmeans;
import numpy as np;
from random import choice;
from matplotlib import pyplot;


def FourFeatures(items, indexA, indexB, indexC, indexD):
    n = len(items);
    X = [];
    for i in range(n):
        item = items[i];
        newItem = [item[indexA], item[indexB], item[indexC], item[indexD] ];
        X.append(newItem);

    return X;


def PlotClusters(clusters):
    n = len(clusters);
    # Cut down the items to two dimension and store to X
    X = [[] for i in range(n)];

    for i in range(n):
        cluster = clusters[i];
        for item in cluster:
            X[i].append(item);

    colors = ['r', 'b', 'g', 'c', 'm', 'y'];

    for x in X:
        # Choose color randomly from list, then remove it
        # (to avoid duplicates)
        c = choice(colors);
        colors.remove(c);

        Xa = [];
        Xb = [];
        Xc = [];
        Xd = [];

        for item in x:
            Xa.append(item[0])
            Xb.append(item[1])
            Xc.append(item[2])
            Xd.append(item[3])

        pyplot.plot(Xa, Xb, Xc, Xd, 'o', color=c);

    pyplot.show();


def main():
    items = kmeans.LoadData('data.txt');
    items = FourFeatures(items, 0, 1, 2, 3)

    k = 3;
    means = kmeans.CalculateMeans(k, items);
    clusters = kmeans.ClusterFinding(means, items);

    PlotClusters(clusters);


main();