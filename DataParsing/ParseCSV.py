import utm
import csv
import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

radius = 6371


def latlngToXY(lat,  lng,  top,  bottom):
    x = radius * lng * math.cos((top[0] + bottom[0]) / 2)
    y = radius * lat
    return [x,  y]


def convertXY(x,  y,  orig):
    x = (x - orig[2])
    y = (y - orig[3])
    return [x,  y]


h = 33.5
w = 41.4
recAngle = math.atan(w / h)
r = math.sqrt(h * h + w * w)


def calcCoordinates(x0,  y0,  alpha):
    x = x0 + r * math.cos(alpha)
    y = y0 + r * math.sin(alpha)
    return x,  y


def convertRadtoDeg(rad):
    return rad * 180 / math.pi


def calcAlpha(x1,  y1,  x2,  y2):
    return math.atan(abs(x1 - x2) / abs(y1 - y2))


if __name__ == '__main__':
    df = pandas.read_csv('./resources/meta.csv')
    origin = [min(df['Latitude']),  min(df['Longitude'])]
    origin.extend(utm.from_latlon(origin[0],  origin[1]))
    top = [max(df['Latitude']),  max(df['Longitude'])]
    top = utm.from_latlon(top[0],  top[1])
    top = convertXY(top[0],  top[1],  origin)
    imagesX = []
    imagesY = []
    patches = []
    for i in range(0,  df.shape[0]):
        xy = utm.from_latlon(df['Latitude'][i],  df['Longitude'][i])
        xy = convertXY(xy[0],  xy[1],  origin)
        imagesX.append(xy[0])
        imagesY.append(xy[1])
        if len(imagesX) == 1:
            patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + math.pi),  h * 2,  w * 2,
                                         edgecolor='b',  facecolor='none',  angle=0.0))
        else:
            prevx = imagesX[i - 1]
            prevy = imagesY[i - 1]
            alph = calcAlpha(prevx,  prevy,  xy[0],  xy[1])
            if prevx < xy[0] and prevy > xy[1]:
                patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + math.pi - alph),  h * 2,  w * 2,
                                             edgecolor='y',  facecolor='none',
                                             angle=-convertRadtoDeg(alph)))
            else:
                if prevx < xy[0] and prevy < xy[1]:
                    patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + math.pi + alph),  h * 2,  w * 2,
                                                 edgecolor='r',  facecolor='none',
                                                 angle=-convertRadtoDeg(alph)))
                else:
                    patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + math.pi),  h * 2,  w * 2,
                                                 edgecolor='b',  facecolor='none',  angle=0.0))
        #     else:
        #         if prevy == xy[0]:
        #             patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + math.pi),  h * 2,  w * 2,
        #                                          edgecolor='b',  facecolor='none',  angle=90.0))
        #         else:
        #             alph = calcAlpha(prevx,  prevy,  xy[0],  xy[1])
        #             if prevx > xy[0]:
        #                 if prevy > xy[1]:
        #                     patches.append(
        #                         plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + alph + math.pi / 2),  h * 2,  w * 2,
        #                                       edgecolor='b',  facecolor='none',
        #                                       angle=-convertRadtoDeg(math.pi - alph)))
        #                 else:
        #                     patches.append(
        #                         plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + (math.pi * 3 / 2) - alph),  h * 2,  w * 2,
        #                                       edgecolor='b',  facecolor='none',
        #                                       angle=convertRadtoDeg(math.pi - alph)))
        #             else:
        #                     if prevy < xy[1]:
        #                         patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle - alph),  h * 2,  w * 2,
        #                                                      edgecolor='b',  facecolor='none',
        #                                                      angle=-convertRadtoDeg(alph)))
        #                     else:
        #                         patches.append(plt.Rectangle(calcCoordinates(xy[0],  xy[1],  recAngle + alph),  h * 2,  w * 2,
        #                                                      edgecolor='b',  facecolor='none',
        #                                                      angle=convertRadtoDeg(alph)))

    # draw graph
    plt.figure(figsize=(20,  1))
    f,  a = plt.subplots()
    a.set_xlabel('metres')
    a.set_ylabel('metres')
    for p in patches:
        a.add_patch(p)

    a.set_xlim(0.0,  top[0])
    a.set_ylim(0.0,  top[1])
    plt.xticks(np.arange(0 - 2 * 40,  int(top[0] + 2 * 40),  20))
    plt.yticks(np.arange(0 - 2 * 40,  int(top[1] + 2 * 40),  20))

    # triple size of the image
    DefaultSize = f.get_size_inches()
    f.set_size_inches(DefaultSize[0] * 3,  DefaultSize[1] * 3)
    f.savefig("./resources/route'.png")
    plt.show()
