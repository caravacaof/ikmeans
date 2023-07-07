import numpy as np
import matplotlib.pyplot as plt


def read_dataset(path):
    data = []
    f = open(path, "r")
    if path in ['data/real/iris.data']:
        for l in f:
            point = []
            for d in l.split(',')[:-1]:
                point.append(float(d))
            data.append(point)
    elif path in ['data/real/isolet.data']:
        for l in f:
            point = []
            for d in l.split(','):
                point.append(float(d))
            data.append(point)
    elif path in ['data/real/HAR.data']:
        for l in f:
            point = []
            for d in l.split():
                point.append(float(d))
            data.append(point)
    elif path in ['data/real/musk.data']:
        for l in f:
            point = []
            for d in l.split()[:-1]:
                point.append(float(d))
            data.append(point)
    elif path in ['data/real/letter-recognition.data']:
        for l in f:
            point = []
            for d in l.split(',')[1:]:
                point.append(float(d))
            data.append(point)
    elif path in ['data/real/KDDCUP04Bio.data']:
        for l in f:
            point = []
            for d in l.split()[1:]:
                point.append(float(d))
            data.append(point)
            print(point)
    elif path in ['data/real/statlog.data']:
        for l in f:
            point = []
            for d in l.split(',')[2:]:
                point.append(float(d))
            data.append(point)
    else:
        for l in f:
            point = [float(l.split()[0]), float(l.split()[1])]
            data.append(point)
    return np.array(data)


def plot_data(data, centroids, dir):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], marker="o", markersize=1, linestyle='None')
    ax.plot(centroids[:, 0], centroids[:, 1], marker="o", markersize=3, linestyle='None', color="red")
    plt.savefig(dir)
