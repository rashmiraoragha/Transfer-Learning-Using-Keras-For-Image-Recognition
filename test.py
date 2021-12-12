import tensorflow as tf
from tensorflow import keras
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

test_path = sys.argv[1]

batch_size=32


test_datagen= ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='training',
    seed = 123)


model.evaluate(test_set)