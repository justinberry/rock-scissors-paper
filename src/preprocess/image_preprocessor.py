import tensorflow as tf

def preprocess_path(path):
  image = tf.io.read_file(path)
  decoded = tf.image.decode_jpeg(image, channels=3)
  resized = tf.image.resize(decoded, [192, 192])
  return resized / 255.0  # normalize to [0,1] range
