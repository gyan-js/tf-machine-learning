import matplotlib.pyplot as plot
from matplotlib.image import imread

from keras.preprocessing.image import ImageDataGenerator



training_data_generator = ImageDataGenerator(
    rotation_range=400,
    fill_mode='nearest'
)
trained_images_directory = "D:/Kunal Programming/PYTHON/tf-machine-learning/damage_detection/Pro-M3-Hurricane-Damage-Dataset/train"

training_augmented_images = training_data_generator.flow_from_directory(
    trained_images_directory,
    target_size=(180, 180)
)

for i in range(4):
    plot.subplot(2, 2, i + 1)
    batch  = training_augmented_images.next()
    image = batch[0][0].astype('uint8')
    plot.imshow(image)

plot.show()
