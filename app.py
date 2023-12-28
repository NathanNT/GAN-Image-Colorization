import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import random

import numpy as np
import os
from PIL import Image

class TestCallback(Callback):
    def __init__(self, test_data, output_folder):
        super().__init__()
        self.test_data = test_data
        self.output_folder = output_folder

    def on_epoch_end(self, epoch, logs=None):
        idx = random.randint(0, len(self.test_data[0]) - 1)
        image_bw, image_color = self.test_data[0][idx], self.test_data[1][idx]
        predicted_color = self.model.predict(np.expand_dims(image_bw, axis=0))

        # Rescale images
        image_bw_display = np.squeeze(image_bw, axis=-1)
        predicted_color_display = np.squeeze(predicted_color, axis=0)

        # Plot and save the image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image_color)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(image_bw_display, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_color_display)
        plt.title('Predicted')
        plt.axis('off')

        plt.savefig(f"{self.output_folder}/epoch_{epoch+1}.png")
        plt.close()


def load_and_preprocess_image(path):
    image = Image.open(path)
    image = image.resize((64, 64))
    image_bw = image.convert('L')  # Convert to grayscale
    image_bw = np.array(image_bw)
    image_bw = np.expand_dims(image_bw, axis=-1)  # Add channel dimension
    image_bw = image_bw / 255.0  # Normalize to [0, 1]

    image_color = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_bw, image_color

def load_dataset(folder):
    images_bw = []
    images_color = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            path = os.path.join(folder, filename)
            image_bw, image_color = load_and_preprocess_image(path)
            images_bw.append(image_bw)
            images_color.append(image_color)
    return np.array(images_bw), np.array(images_color)

train_bw, train_color = load_dataset('train/dog')
test_bw, test_color = load_dataset('test/dog')
def unet_generator(input_size=(64, 64, 1)):
    inputs = Input(input_size)

    # Downsampling
    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), padding='same')(p1)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    bn = Conv2D(256, (3, 3), padding='same')(p2)
    bn = Activation('relu')(bn)

    # Upsampling
    u1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(bn)
    u1 = concatenate([u1, c2])
    u1 = Activation('relu')(u1)

    u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u1)
    u2 = concatenate([u2, c1])
    u2 = Activation('relu')(u2)

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(u2)

    model = Model(inputs, outputs)
    return model
model = unet_generator()
model.compile(optimizer=Adam(lr=0.001), loss='mse')

output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
test_callback = TestCallback(test_data=(test_bw, test_color), output_folder=output_folder)
model.fit(train_bw, train_color, epochs=50, batch_size=32, validation_data=(test_bw, test_color), callbacks=[test_callback])
