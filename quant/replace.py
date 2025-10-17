from .quantized_modules.robuq import init_RobuQLinear_from_Linear
def replace_linear_layer(linear,nbits=4,w_bits=4,if_hadamard=True,if_lora=True):
   return init_RobuQLinear_from_Linear(linear,nbits,w_bits,if_hadamard,if_lora)
