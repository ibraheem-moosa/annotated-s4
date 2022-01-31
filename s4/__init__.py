print('Setting up colab TPU.')
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
from .s4 import *
