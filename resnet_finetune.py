#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

PRE_TRAINED_MODEL_PATH = "/Users/liangyue/Documents/frozen_model_vgg_16/model/resnet_v1_50.ckpt"


class OssPath(object):
    BUCKET_PATH = "oss://yit-prod-pai"
    PREFIX = "ml-pai/image_finetune"
    MODEL_NAME = "resnet_v1_50.ckpt"

    PRE_TRAINED_MODEL_PATH = os.path.join(BUCKET_PATH, PREFIX, "pre_trained_model", MODEL_NAME)
    TF_RECORD_PATH = os.path.join(BUCKET_PATH, PREFIX, "tfrecord_data")
    LOG_DIR_PATH = os.path.join(BUCKET_PATH, PREFIX, "training_log")


def get_trainable_variables(checkpoint_exclude_scopes=None):
    """Return the trainable variables.

    Args:
        checkpoint_exclude_scopes: Comma-separated list of scopes of variables
            to exclude when restoring from a checkpoint.

    Returns:
        The trainable variables.
    """
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in
                      checkpoint_exclude_scopes.split(',')]
    variables_to_train = []
    for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_train.append(var)
    return variables_to_train


def get_record_dataset(record_path,
                       reader=None,
                       num_samples=50000,
                       num_classes=7):
    """Get a tensorflow record file.

    Args:

    """
    if not reader:
        reader = tf.TFRecordReader

    # 将tf.train.Example反序列化成存储之前的格式。由tf完成
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/label': tf.FixedLenFeature((), tf.int64, default_value=0),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                              format_key='image/format',
                                              shape=[224, 224, 3],
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/label'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


from tensorflow.contrib.slim.nets import vgg, resnet_v1


def train(checkpoint_path, record_path, is_oss=False):
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
        net, endpoints = resnet_v1.resnet_v1_50(inputs, None)
        net = tf.squeeze(net, axis=[1, 2])
        logits = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='Predict')
        logits = tf.nn.softmax(logits)

        vars_to_train = get_trainable_variables(
            "resnet_v1_50/conv1, resnet_v1_50/block1, resnet_v1_50/block2, resnet_v1_50/block3, resnet_v1_50/block4"
        )
        # print("vars_to_train")
        # for v in vars_to_train:
        #     print(v, type(v))

        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        slim.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=labels,
            scope='Loss'
        )
        train_op = slim.learning.create_train_op(
            slim.losses.get_total_loss(), optimizer,
            summarize_gradients=True,
            variables_to_train=vars_to_train
        )

        variables_to_restore = []
        for var in slim.get_variables_to_restore():
            variables_to_restore.append(var)

        # print("variables_to_restore from model")
        # for vv in variables_to_restore:
        #     print(vv)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=True)

        if is_oss:
            logdir = OssPath.LOG_DIR_PATH
        else:
            logdir = './training'
        slim.learning.train(train_op=train_op,
                            logdir=logdir,
                            init_fn=init_fn, number_of_steps=10,
                            save_summaries_secs=20,
                            save_interval_secs=600)


def main(_):
    is_oss = False
    if is_oss:
        pre_trained_model_path = OssPath.PRE_TRAINED_MODEL_PATH
        tf_record_fold_path = OssPath.TF_RECORD_PATH
    else:
        pre_trained_model_path = PRE_TRAINED_MODEL_PATH
        tf_record_fold_path = "./tfrecord"
    train(pre_trained_model_path, os.path.join(tf_record_fold_path, "trains-*.tfrecord"))


if __name__ == "__main__":
    tf.app.run(main)
