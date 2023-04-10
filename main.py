import ctypes
import numpy as np
import tensorflow as tf

lib = ctypes.cdll.LoadLibrary("./target/release/librust_tensorflow_bindings.so")
lib.process_strings.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
]
lib.process_strings.restype = None


def tokenize(ragged_tensor):
    result = (ctypes.c_uint * 1000)()
    string_list = ragged_tensor.flat_values.numpy().tolist()
    string_pointers = [ctypes.c_char_p(s) for s in string_list]
    string_pointers_array = (ctypes.c_char_p * len(string_pointers))(*string_pointers)
    lib.process_strings(string_pointers_array, len(string_pointers), result, len(result))
    return np.trim_zeros(np.ctypeslib.as_array(result))

ragged_tensor = tf.ragged.constant([["Hello World"],["Rust", "Python"]])
print(tokenize(ragged_tensor))