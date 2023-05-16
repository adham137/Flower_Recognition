import shutil

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['rose', 'daisy', 'dandelion', 'sunflower', 'tulip']

#moving the images to training and validation arrays
for type in classes:
    images_path = os.path.join(base_dir, type)
    images = glob.glob(images_path + '/*.jpg')
    print("{}: {} images".format(type, len(images)))
    train = images[:round(len(images)*0.8)]
    val = images[round(len(images)*0.8):]

#moving the images in the arrays into files
    for t_image in train:
        if not os.path.exists(os.path.join(base_dir, 'train', type)):
            os.makedirs(os.path.join(base_dir, 'train', type))
        shutil.move(t_image, os.path.join(base_dir, 'train', type))

    for v_image in val:
        if not os.path.exists(os.path.join(base_dir, 'val', type)):
            os.makedirs(os.path.join(base_dir, 'val', type))
        shutil.move(v_image, os.path.join(base_dir, 'val', type))

train_directory = os.path.join(base_dir, 'train')
validate_directory = os.path.join(base_dir, 'val')

#DATA AUGMENTATION
batch_size = 100
IMG_shape = 150

#   for training set
#   applying: random 45 deg rotation,
#             random  zoom, up to 50% zoom,
#             horizontal flip,
#             width and height shift up to 0.15
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            rotation_range=45,
                                                            zoom_range=0.5,
                                                            horizontal_flip=True,
                                                            width_shift_range=0.15,
                                                            height_shift_range=0.15
                                                            )
train_data_gen = image_gen.flow_from_directory(
                                                directory=train_directory,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                target_size=(IMG_shape, IMG_shape),
                                                class_mode='sparse'
                                                )
#   for validation set
#   applying: rescale only
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen.flow_from_directory(
                                                directory=validate_directory,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                target_size=(IMG_shape, IMG_shape),
                                                class_mode='sparse'
                                                )


#CREATING THE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', input_shape=(IMG_shape,IMG_shape,3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=512, activation='relu'),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

#COMPILING THE MODEL
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
#TRAINING THE MODEL
epochs = 80
history = model.fit_generator(
                                generator=train_data_gen,
                                steps_per_epoch=int(np.ceil(train_data_gen.n) / float(batch_size)),
                                epochs=epochs,
                                validation_data=val_data_gen,
                                validation_steps=int(np.ceil(val_data_gen.n) / float(batch_size))
                                )

#PLOTTING THE ACCURACY GRAPHS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




