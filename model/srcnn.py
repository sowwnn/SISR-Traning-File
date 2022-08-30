from tensorflow.keras.layers import Conv2D,Input,UpSampling2D,Lambda
from tensorflow.keras import Model  

import tensorflow as tf 


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)



def upsample(x, scale, num_filters):
  def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

  if scale == 2:
      x = upsample_1(x, 2, name='conv2d_1_scale_2')
  elif scale == 3:
      x = upsample_1(x, 3, name='conv2d_1_scale_3')
  elif scale == 4:
      x = upsample_1(x, 2, name='conv2d_1_scale_2')
      x = upsample_1(x, 2, name='conv2d_2_scale_2')

  return x

def SRCNN(scale,upsamp='pixel_shuffle',num_filter=64):
  input = Input(shape = (None,None,3))

  if (upsamp == 'bicubic'):
    l = UpSampling2D(scale,data_format='channels_last',interpolation="bicubic")(input)
  else: 
    l = upsample(input,scale,num_filter)
  l = Conv2D(64, (5,5), activation = 'relu', padding = 'same')(l)
  # l = BatchNormalization(axis = 3)(l)
  l = Conv2D(32, (1,1), activation = 'relu', padding = 'same')(l)
  # l = BatchNormalization(axis = 3)(l)
  l = Conv2D(3,(3,3), activation = 'relu', padding = 'same')(l)

  output = Conv2D(3,(3,3), activation = 'linear', padding = 'same')(l)
  model = Model(inputs = input, outputs = output, name = 'SRCNN') 
  return model