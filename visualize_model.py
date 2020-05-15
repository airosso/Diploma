import mnist_cnn

if __name__ == '__main__':
    model = mnist_cnn.build_model((70, 70))
    model.summary()
    for layer in model.layers:
        print(layer.input_shape)
