import tensorflow as tf
from tensorflow import keras
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random


train_path = sys.argv[1]
model_name = sys.argv[2]
batch_size=32


train_datagen= ImageDataGenerator(rescale=1./255, 
                                  rotation_range=7, 
                                  width_shift_range=0.5, 
                                  height_shift_range= 0.45, 
                                  shear_range= 0.2, 
                                  zoom_range=0.45,
                                  horizontal_flip=True, 
                                  validation_split=0.2)


train_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='training',
    seed = 123)

val_datagen= ImageDataGenerator(rescale=1./255)

val_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='training',
    seed = 123)

pretrained_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                  input_shape= (224,224,3),
                                                                  weights='imagenet')

layer = pretrained_model.output
layer = tf.keras.layers.Flatten()(layer)
layer = tf.keras.layers.Dense(units=1024, activation='relu')(layer)
layer = tf.keras.layers.Dense(units=512, activation='relu')(layer)
layer = tf.keras.layers.Dense(1)(layer)
out = tf.keras.layers.Activation(activation='sigmoid')(layer)

model=tf.keras.Model(inputs=pretrained_model.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


class model_callbacks(tf.keras.callbacks.Callback):
    def _init_(self,threshold):
        super(model_callbacks, self)._init_()
        self.threshold=threshold
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if (val_acc>= self.threshold):
            self.model.stop_train=True
            
callbacks = model_callbacks(threshold=0.97)

history = model.fit_generator(train_set,
                    epochs = 30, 
                    validation_data=val_set,
                    callbacks=[callbacks])

model.save(model_name)