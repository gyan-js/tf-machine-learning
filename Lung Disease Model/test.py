import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import plot_image
import plot_value

validation_data_generator = ImageDataGenerator()

training_data_generator = ImageDataGenerator(
    # rescale=1.0/255,
    rotation_range=40,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

validation_image_directory = "D:/Kunal Programming/PYTHON/tf-machine-learning/Lung Disease Model/xrayed_images/validation_dataset"

training_image_directory = "D:/Kunal Programming/PYTHON/tf-machine-learning/Lung Disease Model/xrayed_images/training_dataset"
x = np.asarray(validation_image_directory)
validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(180, 180)

)

training_augmented_images = training_data_generator.flow_from_directory(
    training_image_directory,
    target_size=(180, 180))

class_names = list(training_augmented_images.class_indices.keys())
train_labels = training_augmented_images.classes
test_labels = validation_augmented_images.classes

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    
    
    # Flatten the results to feed into a Dense Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    # Classification Layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

predictions = model.predict(validation_augmented_images)


for i in range(9):
    plt.subplot(3, 3, i+1)
    plot_image(i, predictions, test_labels, validation_augmented_images, training_augmented_images)
  
    

plt.tight_layout()
##plt.show()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(training_augmented_images,  epochs=5,
          validation_data=validation_augmented_images)

model.save('pneumothorax.h5')
model.summary()


score = model.evaluate(validation_augmented_images)
print("test Loss:", score[0])
print('Test accuracy:', score[1])
print(training_augmented_images)


for layer in model.layers:
    print(layer.output_shape)

print(class_names)


# https://medium.com/edureka/tensorflow-image-classification-19b63b7bfd95
