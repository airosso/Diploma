import keras
import matplotlib.pyplot as plt
import numpy
import pandas as pandas
import keras.backend as K
import tensorflow as T
from PIL import Image

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc

import format


def mnist_model(train_datasets, test_datasets):
    train_df = images_data_frame(train_datasets)
    test_df = images_data_frame(test_datasets)

    image_datagen = ImageDataGenerator(
        rotation_range=0.2,
        samplewise_center=True,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1. / 65535
    )

    input_shape = (70, 70)

    train_generator = image_datagen.flow_from_dataframe(
        dataframe=train_df,
        target_size=input_shape,
        color_mode='grayscale',
        seed=1
    )

    test_generator = image_datagen.flow_from_dataframe(
        dataframe=test_df,
        target_size=input_shape,
        color_mode='grayscale',
        seed=1
    )

    model = build_model(input_shape)

    model.fit_generator(
        train_generator,
        steps_per_epoch=64,
        epochs=200,
        validation_data=test_generator,
        # use_multiprocessing=True,
        # workers=6,
        verbose=1
    )

    model.save_weights("model_weights.h5")

    score = model.evaluate_generator(
        test_generator,
        verbose=1,
        # workers=6,
        # use_multiprocessing=True
    )
    y_pred = model.predict_generator(test_generator)
    t = y_pred[:, 1:].ravel()
    print(t)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, t)
    auc_keras = auc(fpr_keras, tpr_keras)
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
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(input_shape[0], input_shape[0], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

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
        metrics=[f1_macro]
    )
    return model


def images_data_frame(numbers):
    infos = list(map(format.processed_dataset_info, numbers))
    filenames = []
    classes = []
    for info in infos:
        for a, b in zip(info[0], info[1]):
            filenames.append(a)
            classes.append(b)
            print({a: b})
    return pandas.DataFrame({"filename": filenames, "class": classes})


if __name__ == '__main__':
    mnist_model([4], [3])
