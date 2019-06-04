#%%

import tensorflow as tf
import numpy

AUTOTUNE = tf.data.experimental.AUTOTUNE

from preprocess.image_preprocessor import preprocess_path
from visualisation.image_plotter import plot, plot_all

# When ran in jupyter context will allow resolution
# of our modules
import sys
sys.path.append('./src')

from dataset import factory

image_paths, labelled_images, label_with_index, label_names = factory.create()

img_tensor = preprocess_path(image_paths[0])

print('--------------')
print(img_tensor.shape)
print(img_tensor.dtype)

print(label_with_index)
#plot(img_tensor, labelled_images[0])

path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

image_dataset = path_dataset.map(
  preprocess_path, 
  num_parallel_calls=AUTOTUNE
)

#plot_all(
#  image_dataset=image_dataset,
#  image_paths=image_paths,
#  labelled_images=labelled_images
#)

label_dataset = tf.data.Dataset.from_tensor_slices(
  tf.cast(labelled_images, tf.int64)
) 

#for l in label_dataset.take(10):
#  print(label_names[l.numpy()])

image_label_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

print('image_label_dataset')
print(image_label_dataset)

# Basic methods for training
#
# To train a model with this dataset you will want the data:
#
#    To be well shuffled.
#    To be batched.
#    To repeat forever.
#    Batches to be available as soon as possible.

BATCH_SIZE = 32

# The order is important.
# 
#   A .shuffle after a .repeat would shuffle items across epoch boundaries (some items will be seen twice before others are seen at all).
#   A .shuffle after a .batch would shuffle the order of the batches, but not shuffle the items across batches.

IMAGE_COUNT = len(image_paths)

# batch = image_label_dataset.shuffle(buffer_size=IMAGE_COUNT)
# repeated = batch.repeat()

shuffled_and_repeated = image_label_dataset.cache().apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=IMAGE_COUNT)
)

batched = shuffled_and_repeated.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
the_dataset = batched.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

def normalise_range(image, label):
  return 2 * image - 1, label

# The MobileNet returns a 6x6 spatial grid of features for each image.
keras_dataset = the_dataset.map(normalise_range)

image_batch, label_batch = next(iter(keras_dataset))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

print(len(model.trainable_variables))
model.summary()

steps_per_epoch=tf.math.ceil(len(image_paths) / BATCH_SIZE).numpy()
print(steps_per_epoch)

#model.fit(the_dataset, epochs=50, steps_per_epoch=steps_per_epoch)
model.fit(the_dataset, epochs=1, steps_per_epoch=3)


class_names = sorted(label_with_index.items(), key=lambda pair:pair[1])
class_names = numpy.array([key.title() for key, value in class_names])

predicted_batch = model.predict(the_dataset)
print('predicted_batch')
print(predicted_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
print('predicted_id')
print(predicted_id)
predicted_label_batch = class_names[predicted_id]
print('predicted_label_batch')
print(predicted_label_batch)

label_id = tf.math.argmax(label_batch, axis=-1)
print('label_id')
print(label_id)

import matplotlib.pyplot as matplot

import matplotlib
matplotlib.use('WebAgg')

matplot.figure(figsize=(10,9))
matplot.subplots_adjust(hspace=0.5)
for n in range(30):
  matplot.subplot(6,5,n+1)
  matplot.imshow(image_batch[n])
#  color = "green" if predicted_id[n] == label_id[n] else "red"
  color = "green"
  matplot.title(predicted_label_batch[n].title(), color=color)
  matplot.axis('off')
_ = matplot.suptitle("Model predictions (green: correct, red: incorrect)")
matplot.show()

# 