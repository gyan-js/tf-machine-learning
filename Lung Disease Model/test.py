import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#x_train = x_train.reshape(x_train.shape[0], 30, 30, 1)
#x_test = x_test.reshape(x_test.shape[0], 30, 30, 1)

#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

validation_data_generator = ImageDataGenerator(rescale=1.0/255)

training_data_generator = ImageDataGenerator(
    #rescale=1.0/255,
    rotation_range=40,
    
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

validation_image_directory = "D:/Kunal Programming/PYTHON/data_visualization/image_augmentation/xrayed_images/validation_dataset"

training_image_directory = "D:/Kunal Programming/PYTHON/data_visualization/image_augmentation/xrayed_images/training_dataset"

validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(180, 180))

training_augmented_images = training_data_generator.flow_from_directory(
    training_image_directory,
    target_size=(180, 180))
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(320, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a Dense Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    # Classification Layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(training_augmented_images,  epochs=10,
          validation_data=validation_augmented_images)

model.save('Pneumothorax.h5')
model.summary()
"""

for i in range(4):
    pyplot.subplot(2,2, i+1)
    batch = training_augmented_images.next()
    image = batch[0][0].astype('uint8')
    pyplot.imshow(image)
pyplot.show()
    