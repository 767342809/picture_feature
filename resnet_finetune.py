import tensorflow as tf
import tensorflow.contrib.slim as slim


PRE_TRAINED_MODEL_PATH = "/Users/liangyue/Documents/frozen_model_vgg_16/model/resnet_v1_50.ckpt"


def load_variables_from_model(checkpoint_path: str, is_print_var_name: bool=False):
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    if is_print_var_name:
        for key in var_to_shape_map:
            print("tensor_name: ", key, var_to_shape_map[key])

    return var_to_shape_map


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


def train(checkpoint_path: str, record_path):
    dataset = get_record_dataset(record_path, num_samples=1,
                                 num_classes=1000)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    inputs, labels = tf.train.batch([image, label],
                                    batch_size=1,
                                    allow_smaller_final_batch=True)
    inputs = tf.cast(inputs, tf.float32)

    with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = resnet_v1.resnet_v1_50(inputs, None)
        net = tf.squeeze(net, axis=[1, 2])
        logits = slim.fully_connected(net, num_outputs=1000,
                                      activation_fn=None, scope='resnet_v1_50/logits')
        logits = tf.nn.softmax(logits)
        print(net)

        vars_to_train = get_trainable_variables("resnet_v1_50/logits")
        print("vars_to_train")
        for v in vars_to_train:
            print(v)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.1,
            momentum=0.9
        )

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
        for var in slim.get_model_variables():
            variables_to_restore.append(var)

        print("variables_to_restore from model")
        for vv in variables_to_restore:
            print(vv)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=True)
        # print(init_fn)

        slim.learning.train(train_op=train_op, logdir='./training',
                            init_fn=init_fn, number_of_steps=10,
                            save_summaries_secs=20,
                            save_interval_secs=600)


def image_to_tfrecord(img_file, record_file_num, label, record_path):
    with tf.Session() as sess:
        set_width, set_height = 224, 224
        img_raw = tf.gfile.FastGFile(img_file, 'rb').read()
        decode_data = tf.image.decode_jpeg(img_raw, channels=3)
        decode_data = tf.image.resize_image_with_pad(
            decode_data, target_height=set_height, target_width=set_width, method=tf.image.ResizeMethod.BILINEAR)
        decode_data = tf.cast(decode_data, tf.uint8)
        encoded_image = tf.image.encode_jpeg(decode_data)
        img_raw = encoded_image.eval()

        # 保存图片
        with tf.gfile.GFile(record_path+'resize.jpg', 'wb') as f:
            f.write(encoded_image.eval())

        record_file_name = ("trains-%.3d.tfrecord" % record_file_num)
        writer = tf.python_io.TFRecordWriter(record_path + record_file_name)
        example = tf.train.Example(features=tf.train.Features(feature={
                      'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                      'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
                      'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_width])),
                      'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_height])),
                      'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    tf_record_fold_path = "./tfrecord/"
    image_to_tfrecord("/Users/liangyue/Documents/frozen_model_vgg_16/1.jpg", 0, 1, tf_record_fold_path)
    # train(PRE_TRAINED_MODEL_PATH, tf_record_fold_path + "trains-*.tfrecord")