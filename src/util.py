import tensorflow as tf

def get_mask_for_variable(variable):
    mask = tf.Variable(
        initial_value=tf.ones(variable.shape, dtype=tf.float32),
        trainable=False,
        dtype=tf.float32,
        name="{}_mask".format(variable.name.split(':')[0])
    )
    return mask