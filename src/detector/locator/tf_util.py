import tensorflow as tf


def _Int64Feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _FloatFeature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _BytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def variable_summaries(var, name_scope=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    scope = name_scope or 'name_scope'
    with tf.compat.v1.name_scope(scope):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)
