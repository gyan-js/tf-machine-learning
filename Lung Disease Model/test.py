import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

validation_data_generator = ImageDataGenerator()

training_data_generator = ImageDataGenerator(
    # rescale=1.0/255,
    rotation_range=40,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

validation_image_directory = "D:/Kunal Programming/PYTHON/data_visualization/image_augmentation/xrayed_images/validation_dataset"

training_image_directory = "D:/Kunal Programming/PYTHON/data_visualization/image_augmentation/xrayed_images/training_dataset"
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


for i in range(4):
    plt.subplot(2, 2, i+1)

    predictions = model.predict(validation_augmented_images)
    batch = validation_augmented_images.next()
    image = batch[0][0].astype('uint8')
    predictions, true_label = predictions[i], test_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)

    predicted_labels = np.argmax(predictions)
    if predicted_labels == class_names[0]:
        color = 'red'
    else:
        color = 'green'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_labels],100*np.max(predictions), "."), color=color)
    



plt.show()


'''
for i in range(16):
    plt.subplot(4, 4, i+1)
    batch = training_augmented_images.next()
    image = batch[0][0].astype('uint8')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplots_adjust(hspace=0.3)
    plt.xlabel(class_names[train_labels[0]])
    plt.imshow(image)
plt.show()
'''
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(training_augmented_images,  epochs=1,
          validation_data=validation_augmented_images, batch_size=batch)

model.save('LungDisease.h5')
model.summary()


score = model.evaluate(validation_augmented_images)
print("test Loss:", score[0])
print('Test accuracy:', score[1])
print(training_augmented_images)


for layer in model.layers:
    print(layer.output_shape)

print(class_names)
print(true_label)

# https://medium.com/edureka/tensorflow-image-classification-19b63b7bfd95
