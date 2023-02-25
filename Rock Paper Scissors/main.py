import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

testing_data_generator = ImageDataGenerator(
    fill_mode='nearest'
)
training_data_generator = ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

testing_data_path = '/path/to-directory'
training_data_path = '/path/to-directory'

testing_images = testing_data_generator.flow_from_directory(
    testing_data_path,
    target_size=(180, 180)

)

training_images = training_data_generator.flow_from_directory(
    training_data_path,
    target_size=(180, 180))