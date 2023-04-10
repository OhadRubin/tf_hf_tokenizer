import tensorflow as tf

# Load the compiled shared library containing the custom operation
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

custom_op_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile('custom_op.so'))

# Define a list of input strings
input_strings = [
    "Hello, how are you?",
    "This is a test.",
    "Custom operation in TensorFlow.",
]

# Convert the input strings to a TensorFlow constant tensor
input_tensor = tf.constant(input_strings, dtype=tf.string)

# Call the custom operation 'ProcessStrings'
output_tensor = custom_op_lib.process_strings(input=input_tensor)
print(output_tensor.numpy())
