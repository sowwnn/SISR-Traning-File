#Augmentation (crop, flip, rotate)
from tensorflow.data import AUTOTUNE
import tensorflow as tf 

def read_input(lr,hr):
  lr = tf.io.read_file(lr)
  lr = tf.image.decode_png(lr,channels=3)
  hr = tf.io.read_file(hr)
  hr = tf.image.decode_png(hr,channels=3)
  
  return (lr,hr)


def rd_crop(lr_img, hr_img, hr_crop_size= 48, scale=2):

    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_width = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_height = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_width = lr_width * scale
    hr_height = lr_height * scale

    lr_img_cropped = lr_img[lr_height:lr_height + lr_crop_size, lr_width:lr_width + lr_crop_size]
    hr_img_cropped = hr_img[hr_height:hr_height + hr_crop_size, hr_width:hr_width + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def rd_flip(lr_img, hr_img):

    random = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(random < 0.5,lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def rd_rotate(lr_img, hr_img):

    rd = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rd), tf.image.rot90(hr_img, rd)


def preprocessing(data, batch_size=16,scale=4,repeat_count=None, random_transform=True):

        if random_transform:

            data = data.map(lambda lr, hr: rd_crop(lr, hr, scale=scale),num_parallel_calls=AUTOTUNE)

            data = data.map(rd_rotate,num_parallel_calls=AUTOTUNE)

            data = data.map(rd_flip,num_parallel_calls=AUTOTUNE)

        data = data.batch(batch_size)

        data = data.repeat(repeat_count)

        data = data.prefetch(buffer_size=AUTOTUNE)
        return data