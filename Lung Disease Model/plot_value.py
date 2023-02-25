import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_value(i, predictions, test_labels, class_names ):
    predictions, test_labels = predictions[i], test_labels[i]
    plt.grid(False)
    plt.xticks(range(2), class_names, rotation=25)
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions)  
    thisplot[predicted_label].set_color('red')
    thisplot[test_labels].set_color('green')

sys.modules[__name__] = plot_value
