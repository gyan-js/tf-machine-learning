import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

training_images = ImageDataGenerator(
    rotation_range=40,
    fill_mode='nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
)

path_to_train_data = 'D:/Kunal Programming/PYTHON/tf-machine-learning/Car ML Model/data/training'

training_datagen = training_images.flow_from_directory(
    path_to_train_data,
    target_size=(180, 180),
)

class_names = list(training_datagen.class_indices.keys())
print(class_names)
print(len(class_names))