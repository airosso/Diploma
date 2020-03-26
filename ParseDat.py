import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle


def EnsureDir(FileDir):
    Directory = os.path.dirname(FileDir)
    if not os.path.exists(Directory):
        os.makedirs(Directory)


def LoadPickleData(FileDir):
    with open(FileDir, 'rb') as FileHandler:
        Data = pickle.load(FileHandler)
    return Data


def SavePickleData(Data, FileDir):
    EnsureDir(FileDir)
    with open(FileDir, 'wb') as FileHandler:
        pickle.dump(Data, FileHandler)


switcher = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5
}

data = LoadPickleData('../S3/S3.dat')
print(data)
x = [[], [], [], [], [], []]
y = [[], [], [], [], [], []]
for key in data:
    k = data[key]
    category: str = ""
    for annotation in k:
        if annotation == 'Annotations':
            categoryPackage = k[annotation]
            xi = []
            yi = []
            for image in categoryPackage:
                listXYWH = categoryPackage[image]
                xi.append(listXYWH[0])
                yi.append(listXYWH[1])
            i = switcher.get(category, lambda: "Invalid month")
            x[i].extend(xi)
            y[i].extend(yi)
        else:
            category = k[annotation]


color = ['b', 'g', 'r', 'c', 'm', 'y']
for i in range(0, 6):
    fig, ax = plt.subplots()
    ax.plot(x[i], y[i], color=color[i])
    fig.savefig(str(i) + ".png")
    plt.show()
