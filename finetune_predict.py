import tensorflow as tf
import tensorflow.contrib.slim as slim

from resnet_finetune import *

logdir = './training'
checkpoint_file = tf.train.latest_checkpoint(logdir)
print(checkpoint_file)


def predict(checkpoint_path, record_path):
    num_classes = 1000
    dataset = get_record_dataset(record_path, num_samples=2,
                                 num_classes=num_classes)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    inputs, labels = tf.train.batch([image, label],
                                    batch_size=2,
                                    allow_smaller_final_batch=True)
    inputs = tf.cast(inputs, tf.float32)

    with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = resnet_v1.resnet_v1_50(inputs, None, is_training=False)
        net = tf.squeeze(net, axis=[1, 2])
        logits = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='Predict')
        logits = tf.nn.softmax(logits)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore(), ignore_missing_vars=True)

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([inputs, logits])
        probabilities = probabilities[0, 0:]
        print(probabilities)


predict(checkpoint_file, "./tfrecord/trains-*.tfrecord")