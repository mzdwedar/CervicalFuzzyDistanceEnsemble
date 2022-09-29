import tensorflow as tf

def encode_y(y):
  if(y=='NILM'):
    return 0
  elif(y=='ASC-US'):
    return 1
  elif(y=='ASC-H'):
    return 2
  elif(y=='LSIL'):
    return 3
  elif(y=='HSIL'):
    return 4
  else:
    return 5

def normalize(input_image):
  """normalizes the input image pixel values to be [0,1] """
  input_image = tf.cast(input_image, tf.float32)
  input_image /= 255.0
  return input_image


with tf.device('/cpu:0'):
  @tf.function
  def parse_function(datapoint):
    """
    return a resized and normalized pair of image and mask
    args
      datapoint: a single image and its corresponding segmentation mask

    1. load the image from its path, decode it to jpeg, normalize it to [0,1]
    2. decode the run-length encoding to pixels, then project the mask onto canvas with same size as image
    3. resize both the image and segmentation mask, to math the input size of the network i.e (128,128)
    """
    input_image = tf.io.read_file(datapoint[0])
    input_image = tf.image.decode_image(input_image, expand_animations = False, channels=3)
    input_image = tf.image.resize(input_image, (256, 256), method='nearest')
    input_image = normalize(input_image)

    label = encode_y(datapoint[1])

    return input_image, label