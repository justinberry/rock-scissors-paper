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

def get_label_factory(label_names):
  return lambda index : label_names[index]

image_paths, labelled_images, get_label = factory.create()

img_tensor = preprocess_path(image_paths[0])

path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

image_dataset = path_dataset.map(
  preprocess_path, 
  num_parallel_calls=AUTOTUNE
)

label_dataset = tf.data.Dataset.from_tensor_slices(
  tf.cast(labelled_images, tf.int64)
)

image_label_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

# Basic methods for training
#
# To train a model with this dataset you will want the data:
#
#    To be well shuffled.
#    To be batched.
#    To repeat forever.
#    Batches to be available as soon as possible.

BATCH_SIZE = 64

# The order is important.
# 
#   A .shuffle after a .repeat would shuffle items across epoch boundaries (some items will be seen twice before others are seen at all).
#   A .shuffle after a .batch would shuffle the order of the batches, but not shuffle the items across batches.

IMAGE_COUNT = len(image_paths)

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

NUM_LABELS = 3

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(NUM_LABELS, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'categorical_accuracy'])

# The model is now created and ready to be fit against training set
model.summary()

steps_per_epoch=tf.math.ceil(IMAGE_COUNT / BATCH_SIZE).numpy()

model.fit(the_dataset, epochs=20, steps_per_epoch=steps_per_epoch * 2)
#model.fit(the_dataset, epochs=1, steps_per_epoch=3)

prediction = model.predict(the_dataset)
prediction_indicies = tf.math.argmax(prediction, axis=-1)

import matplotlib.pyplot as matplot

import matplotlib
matplotlib.use('WebAgg')

matplot.figure(figsize=(10,9))
matplot.subplots_adjust(hspace=0.5)
plot_x_width = 8
plot_y_width = 5
for n in range(plot_x_width * plot_y_width):
  matplot.subplot(plot_x_width, plot_y_width, n+1)
  matplot.imshow(image_batch[n])
  matplot.title(get_label(prediction_indicies[n]), color='black')
  matplot.axis('off')
_ = matplot.suptitle('Model predictions')
matplot.show()

# 

#%%
