# CPU Threadpool Runtime in Intel® Extension for TensorFlow\* [Experimental]

Default Python CPU packages compile ITEX against OpenMP oneDNN headers and ship both CPU oneDNN shared libraries. `ITEX_OMP_THREADPOOL` selects the library at load time:

- `ITEX_OMP_THREADPOOL=1` (default): `libonednn_cpu_so.so` (OpenMP).
- `ITEX_OMP_THREADPOOL=0`: `libonednn_cpu_eigen_so.so` (oneDNN THREADPOOL + TensorFlow Eigen).

A THREADPOOL-only backend can also be compiled in with `--define=build_with_threadpool=true`. That build links ITEX kernels against THREADPOOL headers and always loads `libonednn_cpu_eigen_so.so`.

## OpenMP Thread Pool

Use OpenMP when you want oneDNN to own intra-op parallelism via `OMP_NUM_THREADS` / `KMP_*`. The graph initializer then sets `TF_NUM_INTEROP_THREADS` conservatively so OpenMP and TensorFlow inter-op threads do not oversubscribe the machine.

## Using Threadpool Runtime
For workloads with large inter-op concurrency, set `ITEX_OMP_THREADPOOL=0` (or build with `--define=build_with_threadpool=true`). Threadpool runtime supplies non-blocking scheduling between independent operations and better dynamic load balancing.

## Example
Here we show historical comparison examples for different threadpool settings on Intel® Xeon® Platinum 8480+ systems.
1. This example is modified from [a keras example](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/antirectifier.py). 

    ```python
    import tensorflow as tf
    import time
    from tensorflow import keras
    from tensorflow.keras import layers

    """
    ## The Antirectifier layer
    """


    class Antirectifier(layers.Layer):
        def __init__(self, initializer="he_normal", **kwargs):
            super().__init__(**kwargs)
            self.initializer = keras.initializers.get(initializer)

        def build(self, input_shape):
            output_dim = input_shape[-1]
            self.kernel = self.add_weight(
                shape=(output_dim * 2, output_dim),
                initializer=self.initializer,
                name="kernel",
                trainable=True,
            )

        def call(self, inputs):
            inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
            pos = tf.nn.relu(inputs)
            neg = tf.nn.relu(-inputs)
            concatenated = tf.concat([pos, neg], axis=-1)
            mixed = tf.matmul(concatenated, self.kernel)
            return mixed

        def get_config(self):
            # Implement get_config to enable serialization. This is optional.
            base_config = super().get_config()
            config = {"initializer": keras.initializers.serialize(self.initializer)}
            return dict(list(base_config.items()) + list(config.items()))


    """
    ## Let's test-drive it on MNIST
    """

    # Training parameters
    batch_size = 64
    num_classes = 10
    epochs = 20

    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Build the model
    model = keras.Sequential(
        [
            keras.Input(shape=(784,)),
            layers.Dense(256),
            Antirectifier(),
            layers.Dense(256),
            Antirectifier(),
            layers.Dropout(0.5),
            layers.Dense(10),
        ]
    )

    # Compile the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train the model
    # warmup
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, steps_per_epoch=10,  validation_split=0.15)
    # Train the model
    start = time.time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)
    end = time.time()
    print("train time(s): ",end-start)
    # Test the model
    # warmup
    model.evaluate(x_test, y_test)
    start = time.time()
    model.evaluate(x_test, y_test)
    end = time.time()
    print("evaluation time(s): ",end-start)
    ```

   For Eigen thread pool, run with `ITEX_OMP_THREADPOOL=0 numactl -C 0-55 -l python antirectifier.py`, This run takes about 60 seconds to train and 0.26 seconds to evaluate.
   For OpenMP thread pool, run with `OMP_NUM_THREADS=56 KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,verbose,compact,1,0  numactl -C 0-55 -l python antirectifier.py`, This run takes about 85 seconds to train and 0.36 seconds to evaluate. Eigen thread pool is about 30% faster on this small model.

2. Benchmark the inception_v4 example.
    Get the trained model via
    `wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv4_fp32_pretrained_model.pb` before running the following code.
    ```python
    import tensorflow as tf
    from tensorflow.core.framework.graph_pb2 import GraphDef
    import time
    import json
    import os
    
    def load_pb(pb_file):
        with open(pb_file, 'rb') as f:
            gd = GraphDef()
            gd.ParseFromString(f.read())
        return gd
    
    def get_concrete_function(graph_def, inputs, outputs, print_graph=False):
        def imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")
    
        wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
        graph = wrap_function.graph
    
        return wrap_function.prune(
            tf.nest.map_structure(graph.as_graph_element, inputs),
            tf.nest.map_structure(graph.as_graph_element, outputs))
            
    def save_json_data(logs, filename="train.json"):
        with open(filename,"w") as f:
            json.dump(logs,f)
    
    
    def run_infer(concrete_function, shape):
        total_times = 50
        warm = int(0.2*total_times)
        res = []
        for i in range(total_times):
            input_x = tf.random.uniform(shape, minval = 0, maxval= 1.0, dtype=tf.float32)
            bt = time.time()        
            y=concrete_function(input=input_x)
            delta_time = time.time() - bt
            print('Iteration %d: %.3f sec' % (i, delta_time))
            if i >= warm:
                res.append(delta_time)
        latency = sum(res) / len(res)
        return latency
        
    def do_benchmark(pb_file, inputs, outputs, base_shape):
    
        concrete_function = get_concrete_function(
            graph_def=load_pb(pb_file),
            inputs=inputs,
            outputs=outputs,
            print_graph=True)
        bs = 1
        base_shape.insert(0, bs)
        shape = tuple(base_shape)
        latency = run_infer(concrete_function, shape)
        res = {'latency':latency}
    
    
        print("Benchmark is done!")
        print("Benchmark res {}".format(res))
        print("Finished")
        return res
        
    def benchmark():
        pb_file = "inceptionv4_fp32_pretrained_model.pb"
        inputs = ['input:0']
        outputs = ['InceptionV4/Logits/Predictions:0']
        base_shape = [299, 299, 3]
    
        return do_benchmark(pb_file, inputs, outputs, base_shape)
    if __name__ == "__main__":
        benchmark()
    ```
   For Eigen thread pool, run with `ITEX_OMP_THREADPOOL=0 numactl -C 0-3 -l python benchmark.py`. The latency is about 0.08 second/image.
    For OpenMP thread pool, run with `OMP_NUM_THREADS=4 KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,verbose,compact,1,0  numactl -C 0-3 -l python benchmark.py`. The latency is 0.04 second/image. OMP thread pool is about 2x slower than Eigen thread pool on this model.
