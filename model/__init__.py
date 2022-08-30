from model import edsr
from model import srcnn


def get_generator(model_arc, is_train=True):
  if model_arc == 'edsr':
    return(edsr.gen)