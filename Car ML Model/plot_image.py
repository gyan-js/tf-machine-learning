import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_image(i, predictions, true_label, test_images, train_images ):
    true_label, predictions = true_label[i], predictions[i]
    class_names = list(train_images.class_indices.keys())
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    batch = test_images.next()
    image = batch[0][0].astype('uint8')
    plt.imshow(image)

    predicted_label = np.argmax(predictions)

    if predicted_label == true_label:
        color = 'red'
       
    else:
        color = 'green'
        

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions), "."))
    


sys.modules[__name__] = plot_image
