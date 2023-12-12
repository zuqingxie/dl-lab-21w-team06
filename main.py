import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(physical_devices))

if physical_devices:
    # Choose the first GPU (assuming you have one)
    gpu = physical_devices[0]
    print("Running a simple GPU test...")
    with tf.device(gpu.name):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(c)