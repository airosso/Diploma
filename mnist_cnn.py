import pandas as pandas
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
import keras

import format


def mnist_model(train_datasets, test_datasets):
    train_df = images_data_frame(train_datasets)
    test_df = images_data_frame(test_datasets)

    image_datagen = ImageDataGenerator(
        rotation_range=0.2,
        samplewise_center=True,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True
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
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=64,
        epochs=20,
        validation_data=test_generator,
        # use_multiprocessing=True,
        # workers=6,
        verbose=1
    )

    score = model.evaluate_generator(
        test_generator,
        verbose=1,
        # workers=6,
        # use_multiprocessing=True
    )
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


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
