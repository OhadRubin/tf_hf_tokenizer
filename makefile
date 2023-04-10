cargo build --release
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared custom_op.cc -o custom_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -L/home/ohadr/rust_tensorflow_bindings/target/release/ -lrust_library
export LD_LIBRARY_PATH="/home/ohadr/rust_tensorflow_bindings/target/release/:$LD_LIBRARY_PATH"
python3 main.py 