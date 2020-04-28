import codecs
import os
import pickle

import tqdm
from PIL import Image


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
                    print("Missing %s" % image_file)
                    missing_files += 1
        print("Missing %d files in R%d" % (missing_files, number))
        return images, classes


def crop_image(image, square, out):
    file = Image.open(image)
    left, top, w, h = square
    file.crop((left, top, left + w, top + h)).save(out)


def process_datasets(numbers, folder):
    infos = list(map(dataset_info, numbers))
    if not os.path.exists(folder):
        os.mkdir(folder)
    for number, info in zip(numbers, infos):
        dataset_path = os.path.join(folder, "R%d" % number)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        images, squares, categories, ids, missing_files = info
        print("Dataset %d missing %d files" % (number, len(missing_files)))
        for image, square, category, key in tqdm.tqdm(zip(images, squares, categories, ids),
                                                      desc="Format dataset S%d" % number):
            crop_image(image, square, os.path.join(dataset_path, "%s_%s.tiff" % (category, key)))


if __name__ == '__main__':
    process_datasets([3, 4], "processed_data")
