import tensorflow as tf
import numpy as np
from PIL import Image

def openTraining():
    fImages = open("MNIST-train-images", "rb")
    fLabels = open("MNIST-train-labels", "rb")
    fImages.read(16)
    fLabels.read(8)

    retImages = np.ndarray((60000, 28, 28, 1), dtype=np.uint8)
    retLabels = np.ndarray((60000), dtype=np.uint8)

    for i in range(60000):
        retImages[i] = np.reshape(np.frombuffer(fImages.read(28 * 28), dtype=np.uint8), (28,28,1))
        retLabels[i] = fLabels.read(1)[0]

    return ((retImages/255.).astype(np.float32), retLabels)

def openTest():
    fImages = open("MNIST-test-images", "rb")
    fLabels = open("MNIST-test-labels", "rb")
    fImages.read(16)
    fLabels.read(8)

    retImages = np.ndarray((10000, 28, 28, 1), dtype=np.uint8)
    retLabels = np.ndarray((10000), dtype=np.uint8)

    for i in range(10000):
        retImages[i] = np.reshape(np.frombuffer(fImages.read(28 * 28), dtype=np.uint8), (28,28,1))
        retLabels[i] = fLabels.read(1)[0]

    return ((retImages/255.).astype(np.float32), retLabels)


def trainModel(images, labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=3),
        tf.keras.layers.Conv2D(48, 3, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(tf.keras.optimizers.Adagrad(.1), 'sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(images, labels, 64, 5)

    return model


if __name__ == "__main__":
    img = np.array(Image.open("test_input.bmp").getchannel("R"))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)


    print("Opening training data...")
    (images, labels) = openTraining()
    print("Training data opened")

    print("Training model...")
    model = trainModel(images, labels)
    print("Model trained")

    print("Opening test data...")
    (images, labels) = openTest()
    print("Test data opened")

    print("Evaluating model on test data...")
    loss, acc = model.evaluate(images, labels)
    print("Acc: ", acc)

    print("This image contains a ", model.predict((img/255.).astype(np.float32), 1).argmax())