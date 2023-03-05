import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import plot_image

validation_images = ImageDataGenerator()

training_images = ImageDataGenerator(
    # rescale=1.0/255,
    rotation_range=40,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

path_to_val = "D:/Kunal Programming/PYTHON/tf-machine-learning/Car ML Model/data/train"

path_to_train = "D:/Kunal Programming/PYTHON/tf-machine-learning/Car ML Model/data/val"
x = np.asarray(path_to_val)
validation_datagen = validation_images.flow_from_directory(
    path_to_val,
    target_size=(180, 180)

)

training_datagen = training_images.flow_from_directory(
    path_to_train,
    target_size=(180, 180))

class_names = list(training_datagen.class_indices.keys())
train_labels = training_datagen.classes
test_labels = validation_datagen.classes

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
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
    tf.keras.layers.Dense(7, activation='softmax')
])

predictions = model.predict(validation_datagen)


for i in range(9):
    plt.subplot(3, 3,i+1)
    plot_image(i, predictions, test_labels, validation_datagen, training_datagen)

    


plt.show()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(training_datagen,  epochs=1,
          validation_data=validation_datagen)

model.save('animal_face.h5')
model.summary()


score = model.evaluate(validation_datagen)
print("Test Loss:", score[0])
print('Test accuracy:', score[1]*100)
print(training_datagen)


for layer in model.layers:
    print(layer.output_shape)

print(class_names)
