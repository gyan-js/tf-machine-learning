from matplotlib import pyplot

from matplotlib.image import imread

infected_testing_image = 'D:/Kunal Programming/Python/data_visualization/image_augmentation/xrayed_images/testing_dataset/infected/testing_image_1.png'

image = imread(infected_testing_image)

pyplot.title('Testing Image')

pyplot.imshow(image)





