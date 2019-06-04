import matplotlib.pyplot as matplot

import matplotlib
matplotlib.use('WebAgg')

def plot(image, label):
  matplot.imshow(image)
  matplot.grid(False)
  matplot.xlabel(label)
  matplot.title('all the things')
  matplot.show()
  print()

def plot_all(image_dataset, image_paths, labelled_images):
  matplot.figure(figsize=(8,8))
  for n, image in enumerate(image_dataset.take(4)):
    matplot.subplot(2,2,n+1)
    matplot.imshow(image)
    matplot.grid(False)
    matplot.xticks([])
    matplot.yticks([])
    matplot.xlabel(image_paths[n] + ' - ' + str(labelled_images[n]))
    matplot.title('all the things')
  matplot.show()