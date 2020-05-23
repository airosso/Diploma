import pickle

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy
import pandas as pandas
import tensorflow as T
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import ResNet50

import format


def mnist_model(train_datasets, test_datasets):
    train_df = images_data_frame(train_datasets + test_datasets)
    train, test = train_test_split(train_df, test_size=0.2)

    with open("mean_std.pickle", "rb") as file:
        mean, std = pickle.load(file)

    print(mean, std)

    def preprocess(image):
        return numpy.divide(numpy.subtract(numpy.divide(image, 65535.0), mean), std)

    image_datagen = ImageDataGenerator(
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess,
        dtype='uint16'
    )

    input_shape = (150, 150)

    train_generator = image_datagen.flow_from_dataframe(
        dataframe=train,
        target_size=input_shape,
        color_mode='grayscale',
        seed=1
    )

    test_generator = image_datagen.flow_from_dataframe(
        dataframe=test,
        target_size=input_shape,
        color_mode='grayscale',
        seed=1
    )

    model = build_model(input_shape)

    model.load_weights("with_surroundings.h5")

    model.fit_generator(
        train_generator,
        steps_per_epoch=64,
        epochs=500,
        validation_data=test_generator,
        callbacks=[keras.callbacks.ModelCheckpoint("model_weights.h5")],
        # use_multiprocessing=True,
        # workers=6,
        # verbose=1
    )

    # model.save_weights("model_weights.h5")

    score = model.evaluate_generator(
        test_generator,
        verbose=1,
        # workers=6,
        # use_multiprocessing=True
    )

    def less_05(d):
        if d[0] > d[1]:
            return 0
        else:
            return 1

    def match(x):
        if x[0] == x[1]:
            return x[2]
        else:
            return None

    def falsePositive(x):
        if x[0] == 1 and x[1] == 0:
            return x[2]
        else:
            return None

    def falseNegative(x):
        if x[0] == 0 and x[1] == 1:
            return x[2]
        else:
            return None

    filterNone = lambda x: x is not None

    generator = model.predict_generator(test_generator)
    y_pred = list(map(less_05, generator))
    matched = list(filter(filterNone, map(match, zip(y_pred, test_generator.classes, test_generator.filenames))))
    falsePositives = list(
        filter(filterNone, map(falsePositive, zip(y_pred, test_generator.classes, test_generator.filenames))))
    falseNegatives = list(
        filter(filterNone, map(falseNegative, zip(y_pred, test_generator.classes, test_generator.filenames))))

    print("Total negative %d" % len(list(filter(lambda x: x == 0, test_generator.classes))))
    print("Total positive %d" % len(list(filter(lambda x: x == 1, test_generator.classes))))
    print("Matched %d" % len(matched))
    print("FP %d" % len(falsePositives))
    print("FN %d" % len(falseNegatives))

    with open("res/matched.pickle", "wb") as file:
        pickle.dump(matched, file)
    with open("res/fp.pickle", "wb") as file:
        pickle.dump(falsePositives, file)
    with open("res/fn.pickle", "wb") as file:
        pickle.dump(falseNegatives, file)

    t = generator[:, 1:].ravel()
    print(t)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, t)
    auc_keras = auc(fpr_keras, tpr_keras)
    print(auc_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='CNN (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def build_model(input_shape):
    image_input = keras.Input((input_shape[0], input_shape[0], 1))
    # coordinates_input = keras.Input((2,))

    model = ResNet50()
    model.load_weights("transfer.h5")
    transfer_input = keras.Input((model.input_shape[0], model.input_shape[0], 1))

    conv_2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    )(image_input)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropout_1 = Dropout(0.25)(max_pool_1)
    conv_2 = Conv2D(64, (3, 3), activation='relu')(dropout_1)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropout_2 = Dropout(0.25)(max_pool_2)
    conv_3 = Conv2D(128, (3, 3), activation='relu')(dropout_2)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    dropout_3 = Dropout(0.5)(max_pool_3)
    flatten = Flatten()(dropout_3)
    together = flatten
    keras.layers.concatenate([flatten, model.layers[-1].output])
    dense_1 = Dense(128, activation='relu')(together)
    dropout_4 = Dropout(0.5)(dense_1)
    dense_2 = Dense(2, activation='softmax')(dropout_4)

    model = keras.Model(
        inputs=[image_input, transfer_input],
        outputs=[dense_2]
    )

    def f1_macro(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = T.where(T.math.is_nan(f1), T.zeros_like(f1), f1)
        return K.mean(f1)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=[f1_macro, 'accuracy']
    )
    return model


def transfer():
    model = ResNet50()

    for l in model.layers[:len(model.layers) / 2]:
        l.trainable = False

    train_df = images_data_frame([3, 4])
    train, test = train_test_split(train_df, test_size=0.2)

    with open("mean_std.pickle", "rb") as file:
        mean, std = pickle.load(file)

    print(mean, std)

    def preprocess(image):
        return numpy.divide(numpy.subtract(numpy.divide(image, 65535.0), mean), std)

    image_datagen = ImageDataGenerator(
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess,
        dtype='uint16'
    )

    input_shape = (150, 150)

    train_generator = image_datagen.flow_from_dataframe(
        dataframe=train,
        target_size=input_shape,
        color_mode='grayscale',
        seed=1
    )


    test_generator = image_datagen.flow_from_dataframe(
        dataframe=test,
        target_size=input_shape,
        color_mode='grayscale',
        seed=1
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=64,
        epochs=500,
        validation_data=test_generator,
        callbacks=[keras.callbacks.ModelCheckpoint("transfer.h5")],
    )


def images_data_frame(numbers):
    infos = list(map(format.processed_dataset_info, numbers))
    filenames = []
    classes = []
    true_leak = 0
    false_leak = 0
    for info in infos:
        for a, b in zip(info[0], info[1]):
            filenames.append(a)
            classes.append(b)
            if b == 'True leak':
                true_leak += 1
            else:
                false_leak += 1
    print("True: " + str(true_leak))
    print("False: " + str(false_leak))
    return pandas.DataFrame({"filename": filenames, "class": classes})


if __name__ == '__main__':
    mnist_model([4], [3])
