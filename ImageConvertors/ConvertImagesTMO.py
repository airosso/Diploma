import numpy as np
import math
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import tifffile
from PIL import Image

# x - np.array
from ConvertImagesHOT import convert2Hot


def TMONorm(x):
    h = x.shape[0]
    y = x.astype(float)
    y = np.ravel(y)
    y = y - min(y)
    y = y / max(y)
    y = 255 * y
    index = 0
    for i in y:
        if i > 255:
            y[index] = 255
        else:
            if i < 0:
                y[index] = 0
    index = index + 1
    return (y.reshape(h, -1)).astype(np.uint8)


def TMOMai(x):
    vmax = 255
    delta = 0.1
    h = x.shape[0]
    l = x.astype(np.uint16)
    l = np.ravel(l)
    l = l.astype(float)
    l = l - min(l) + 1
    l = np.log10(l)
    k = math.floor(max(l) / delta) + 1
    P = np.zeros(k)
    S = np.zeros(k)
    V = np.zeros(k)
    LS = np.zeros(k)

    for i in range(len(l)):
        j = math.floor(l[i] / delta)
        P[j] = P[j] + 1
    P = P / len(l)
    P = np.power(P, 1 / 3)
    SPD = delta * sum(P)

    for i in range(k):
        S[i] = vmax * P[i] / SPD
    vec = np.zeros(len(l))

    vprev = 0
    for i in range(len(V)):
        V[i] = delta * S[i] + vprev
        vprev = V[i]

    lprev = 0
    maxl = max(l)
    for i in range(len(LS)):
        LS[i] = min(maxl, delta + lprev)
        lprev = LS[i]

    for i in range(len(l)):
        j = math.floor(l[i] / delta)
        vec[i] = (l[i] - LS[k]) * S[k] + V[k]

    for i in range(len(vec)):
        if i < 0:
            i = 0
        if i > 255:
            i = 255
    return (vec.reshape(h, -1)).astype(np.uint8)


def TMODrago(x):
    h = x.shape[0]
    Y = x.astype(np.uint16)
    Y = np.ravel(Y)
    Y = Y.astype(float)
    Y = Y - min(Y) + 1
    Ldmax = 255.0
    Lwmax = max(Y)
    b = 0.85
    biasP = math.log(b) / math.log(0.5)
    Z = np.zeros(len(Y))

    for i in range(len(Y)):
        a2 = math.log(Y[i] + 1)
        a3 = math.log10(Lwmax + 1)
        a4 = math.log(2 + 8 * np.power((Y[i] / Lwmax), biasP))
        Z[i] = Ldmax * a2 / (a3 * a4)
    for i in range(len(Z)):
        if i < 0:
            i = 0
        if i > 255:
            i = 255

    return (Z.reshape(h, -1)).astype(np.uint8)


def TMOReinhard(x):
    h = x.shape[0]
    newx = x.astype(np.uint16)
    newx = np.ravel(newx)
    m0 = 0.3
    m1 = 0.7
    m2 = 1.4

    maxin = 256
    maxi = pow(2, 16) - 1
    lav = 0.0
    llav = 0.0

    y = float(newx[1]) / float(maxi)
    minl = math.log(y)
    maxl = math.log(y)
    miny = y
    maxy = y
    cnt = 0

    for xi in newx:
        y = float(xi) / float(maxi)
        if y > 0.0:
            logy = math.log(y)
            lav = lav + 1
            llav = llav + logy
            if logy < minl:
                minl = logy
            if logy > maxl:
                maxl = logy
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
            cnt = cnt + 1

    lav = lav / cnt
    llav = llav / cnt
    if maxl <= minl:
        m = m0
    else:
        k = (maxl - llav) / (maxl - minl)
        if k > 0.0:
            m = m0 + np.power(m1 * k, m2)
        else:
            m = m0

    tonemapping = []
    for i in range(maxin):
        out = float(i) / float(maxin)
        inn = maxi * (miny + (maxy - miny) * pow(lav, m) * out / (1.0 - out))
        if inn < 0.0:
            inn = 0.0
        if inn > maxi:
            inn = maxi
        tonemapping.append(math.floor(inn))

    outmax = pow(2, 16) - 1
    inmax = pow(2, 8) - 1
    print("tonemapping: ", len(tonemapping), "inmax:", inmax)
    output = np.zeros(outmax + 1)
    lastj = inmax
    lastanchor = inmax
    lastfilled = 0
    j = inmax
    last = tonemapping[j]

    if last < round((outmax + 1) * 3 / 4):
        last = outmax

    while 1:
        current = tonemapping[j]
        if current == last:
            output[last] = (lastanchor + j) >> 1
            lastfilled = 1
        else:
            if last > current:
                mid = ((current + last + 1) >> 1) - 1
            else:
                mid = ((current + last - 1) >> 1) - 1
            while last != mid:
                if lastfilled == 0:
                    output[last] = lastj
                if last > mid:
                    last = last - 1
                else:
                    last = last + 1
                lastfilled = 0

            while last != current:
                if lastfilled == 0:
                    output[last] = lastj
                if last > current:
                    last = last - 1
                else:
                    last = last + 1
                lastfilled = 0
            lastfilled = j
        lastj = j
        last = current
        j = j - 1
        if j == -1:
            break

    if not lastfilled:
        output[last] = lastj

    if outmax > 4:
        i1 = output[0]
        i2 = output[1]
        i3 = output[2]

        if i1 > i2:
            d1 = i1 - i2
        else:
            d1 = i2 - i1

        if i3 > i2:
            d2 = i3 - i2
        else:
            d2 = i2 - i3

        if d1 > 2 * d2:
            output[0] = 2 * i2 - i3

        i1 = output[outmax]
        i2 = output[outmax - 1]
        i3 = output[outmax - 2]

        if i1 > i2:
            d1 = i1 - i2
        else:
            d1 = i2 - i1

        if i3 > i2:
            d2 = i3 - i2
        else:
            d2 = i2 - i3

        if d1 > 2 * d2:
            output[outmax] = 2 * i2 - i3

    LDR = np.zeros(len(newx))
    for i in range(len(newx)):
        LDR[i] = output[newx[i]]
    return (LDR.reshape(-1, h)).astype(np.uint8)


def sum4Arrays(x1, x2, x3, x4):
    ans = np.array([[0] * 2 * x1.shape[1] for _ in range(2 * x1.shape[0])])
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            ans[i][j] = x1[i][j]
        for j in range(x2.shape[1]):
            ans[i][j + x1.shape[1]] = x2[i][j]
    for i in range(x3.shape[0]):
        for j in range(x3.shape[1]):
            ans[i + x1.shape[0]][j] = x3[i][j]
        for j in range(x4.shape[1]):
            ans[i + x1.shape[0]][j + x3.shape[1]] = x4[i][j]
    return ans


def sum4YArrays(x1, x2, x3, x4):
    h = x1.shape[0]
    w = x1.shape[1]
    s = (h * 2, w * 2, 3)
    ans = np.zeros(s)
    for k in range(3):
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                ans[i][j][k] = x1[i][j][k]
            for j in range(x2.shape[1]):
                ans[i][j + x1.shape[1]][k] = x2[i][j][k]
        for i in range(x3.shape[0]):
            for j in range(x3.shape[1]):
                ans[i + x1.shape[0]][j][k] = x3[i][j][k]
            for j in range(x4.shape[1]):
                ans[i + x1.shape[0]][j + x3.shape[1]][k] = x4[i][j][k]
    return ans


if __name__ == '__main__':
    x = tifffile.imread("./resources/2018.01.06_I0-0_DC_NUCF_0821.tiff")
    # x = tifffile.imread("./resources/PHOTO_2020_03_12_21_38_57.tiff")
    x_norm = TMONorm(x)
    x_mai = TMOMai(x)
    x_drago = TMODrago(x)
    x_rainhard = TMOReinhard(x)
    img = Image.fromarray(x_norm)
    img.show()
    img = Image.fromarray(x_mai)
    img.show()
    img = Image.fromarray(x_drago)
    img.show()
    img = Image.fromarray(x_rainhard)
    img.show()