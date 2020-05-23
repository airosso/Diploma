import codecs
import os
import pickle
import sys

import numpy
import pandas
import tqdm
import utm
from PIL import Image, ImageOps, ImageStat
from PIL.ImageDraw import ImageDraw

from DataParsing.ParseCSV import convertXY
from libtiff import TIFF


def dataset_info(number):
    missing_files = []
    with codecs.open("data/S%d/dat.dat" % number, mode="rb") as file:
        data = pickle.load(file)
        images, squares, categories, ids = [], [], [], []
        for key, d in data.items():
            category = d["Category"]
            for image, coordinates in d["Annotations"].items():
                parent = image.rsplit("_", 1)[0] + "_5-12-8"
                image_file = "data/S%d/%s/%s" % (number, parent, image)
                if os.path.exists(image_file):
                    images.append(image_file)
                    squares.append(coordinates)
                    categories.append(category)
                    ids.append(str(key) + "_" + image[:-5])
                else:
                    missing_files.append(image_file)
        return images, squares, categories, ids, missing_files


def processed_dataset_info(number):
    dataset_folder = "data/S%d" % number
    # fileToXY = {}
    # for folder in os.listdir(dataset_folder):
    #     meta = os.path.join(folder, 'meta.csv')
    #     df = pandas.read_csv(meta)
    #     origin = [min(df['Latitude']), min(df['Longitude'])]
    #     origin.extend(utm.from_latlon(origin[0], origin[1]))
    #     top = [max(df['Latitude']), max(df['Longitude'])]
    #     top = utm.from_latlon(top[0], top[1])
    #     top = convertXY(top[0], top[1], origin)
    #     for i in range(0, df.shape[0]):
    #         file = df['ImageName'][i]
    #         xy = utm.from_latlon(df['Latitude'][i], df['Longitude'][i])
    #         xy = convertXY(xy[0], xy[1], origin)
    #         assert fileToXY[file] is None
    #         fileToXY[file] = xy

    with codecs.open("data/S%d/dat.dat" % number, mode="rb") as file:
        data = pickle.load(file)
        missing_files = 0
        images, classes = [], []
        for key, d in data.items():
            category = d["Category"]
            for image, coordinates in d["Annotations"].items():
                image = category + "_" + str(key) + "_" + image
                image_file = "processed_data/R%d/%s" % (number, image)
                if os.path.exists(image_file):
                    images.append(image_file)
                    if category in ["A", "B", "E"]:
                        classes.append("True leak")
                    else:
                        classes.append("False leak")
                else:
                    # print("Missing %s" % image_file)
                    missing_files += 1
        # print("Missing %d files in R%d" % (missing_files, number))
        return images, classes


def crop_image(image, square, out):
    t = TIFF.open(image, mode='r')
    arr = t.read_image()
    t.close()
    left, top, w, h = square
    extend = (w + h) // 2
    left_1 = max(left - extend, 0)
    top_1 = max(top - extend, 0)
    right_1 = min(left + w + extend, arr.shape[1])
    bottom_1 = min(top + h + extend, arr.shape[0])
    arr1 = arr[top_1:bottom_1, left_1:right_1]
    arr2 = numpy.copy(arr).astype('float')
    arr2[top:(top + h), left:(left + w)] = 1.0
    centerY = top + h / 2
    centerX = left + w / 2
    R1 = ((w / 2) ** 2 + (h / 2) ** 2) ** 0.5
    R2 = (w ** 2 + h ** 2) ** 0.5
    for i in range(top_1, bottom_1):
        for j in range(left_1, right_1):
            d = ((i - centerY) ** 2 + (j - centerX) ** 2) ** 0.5
            if d < R1:
                v = 1.0
            else:
                v = max((1 - (d - R1) / (R2 - R1)), 0) ** 0.8
            arr2[i][j] = v
    arr2 = arr2[top_1:bottom_1, left_1:right_1]
    # arr3 = numpy.multiply(arr1, arr2).astype('uint16')
    t = TIFF.open(out, 'w')
    t.write_image(arr1)
    t.close()


def normalize(arr):
    arr = arr.astype('float')
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (65535.0 / (maxval - minval))
    return arr


def draw_rectangle_image(image, square, out):
    file = Image.open(image)
    draw = ImageDraw(file)
    left, top, w, h = square
    draw.line([(left, top), (left + w, top), (left + w, top + h), (left, top + h), (left, top)], fill="red")
    file.save(out, "JPEG")


def process_datasets(numbers, folder):
    infos = list(map(dataset_info, numbers))
    if not os.path.exists(folder):
        os.mkdir(folder)
    # mean = dataset_mean(numbers)
    # std = dataset_std(numbers, mean)
    # with open("mean_std.pickle", "wb") as file:
    #     pickle.dump((mean, std), file)
    for number, info in zip(numbers, infos):
        dataset_path = os.path.join(folder, "R%d" % number)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        images, squares, categories, ids, missing_files = info
        print("Dataset %d missing %d files" % (number, len(missing_files)))
        for image, square, category, key in tqdm.tqdm(zip(images, squares, categories, ids),
                                                      desc="Format dataset S%d" % number):
            crop_image(image, square, os.path.join(dataset_path, "%s_%s.tiff" % (category, key)))


def dataset_mean(numbers):
    infos = list(map(dataset_info, numbers))
    mean = 0.0
    count = 0
    for number, info in zip(numbers, infos):
        for image in tqdm.tqdm(info[0], desc="Calc mean dataset S%d" % number):
            mean += image_mean(image)
            count += 512 * 640
    return mean / count


def dataset_std(numbers, mean):
    infos = list(map(dataset_info, numbers))
    std = 0.0
    count = 0
    for number, info in zip(numbers, infos):
        for image in tqdm.tqdm(info[0], desc="Calc std dataset S%d" % number):
            std += image_std(image, mean)
            count += 512 * 640
    return (std / count) ** 0.5


def image_mean(image):
    return numpy.divide(numpy.array(Image.open(image)).astype('float'), 65535).sum()


def image_std(image, mean):
    return numpy.power(numpy.subtract(numpy.divide(numpy.array(Image.open(image)).astype('float'), 65535), mean), 2).sum()


def reformat_results():
    with open("res/matched.pickle", "rb") as file:
        matched = pickle.load(file)
    with open("res/fp.pickle", "rb") as file:
        falsePositives = pickle.load(file)
    with open("res/fn.pickle", "rb") as file:
        falseNegatives = pickle.load(file)

    def copyTo(files, folder):
        from shutil import copyfile
        if not os.path.exists(folder):
            os.mkdir(folder)
        for file in files:
            newFile = os.path.join(folder, file.split('/')[-1])
            copyfile(file, newFile)

    copyTo(matched, "res/matched")
    copyTo(falsePositives, "res/fp")
    copyTo(falseNegatives, "res/fn")


if __name__ == '__main__':
    reformat_results()
    # process_datasets([3, 4], "processed_data")
