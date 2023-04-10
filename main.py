import ctypes
import numpy as np
import tensorflow as tf

# Load the Rust shared library
# lib = ctypes.cdll.LoadLibrary("./liblib.so")
lib = ctypes.cdll.LoadLibrary("./target/release/librust_tensorflow_bindings.so")

# Rust function signatures
# lib.process_strings.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
lib.process_strings.restype = None
# lib.free_string.argtypes = [ctypes.c_char_p]
# lib.free_string.restype = None

# Create a Ragged string tensor in Python
ragged_tensor = tf.ragged.constant([["Hello World"],["Rust", "Python"]])
print(tf.shape(ragged_tensor))
# Convert the Ragged string tensor to a flat list of strings
string_list = ragged_tensor.flat_values.numpy().tolist()

# Convert the list of strings to a format suitable for passing to Rust
string_pointers = [ctypes.create_string_buffer(s) for s in string_list]
string_pointers_array = (ctypes.c_char_p * len(string_pointers))(*[ctypes.addressof(s) for s in string_pointers])

# Call the Rust function
# lib.process_strings(string_pointers_array, len(string_pointers))
lib.process_strings.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
]

# Allocate memory for the result array (change 1000 to the desired maximum size)
result = (ctypes.c_uint * 1000)()

# Call the Rust function
lib.process_strings(string_pointers_array, len(string_pointers), result, len(result))

# Convert the result to a Python list
token_ids = [result[i] for i in range(len(result)) if result[i] != 0]

# Print the token IDs
print(token_ids)


