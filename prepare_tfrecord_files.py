import os
import ssl
import oss2
import urllib.request
import tensorflow as tf

from Constant import OssConfig, OssPath

ssl._create_default_https_context = ssl._create_unverified_context

LOCAL_TFRECORD_PATH = "./tfrecord"


def image_to_tfrecord(img_file, label, image_id, local_tf, is_save_show_picture=False):
    with tf.Session():
        set_width, set_height = 224, 224
        if img_file.startswith("https:"):
            req = urllib.request.Request(img_file)
            response = urllib.request.urlopen(req)
            img_raw = response.read()
        else:
            img_raw = tf.gfile.FastGFile(img_file, 'rb').read()
        decode_data = tf.image.decode_jpeg(img_raw, channels=3)
        decode_data = tf.image.resize_image_with_pad(
            decode_data, target_height=set_height, target_width=set_width, method=tf.image.ResizeMethod.BILINEAR)
        decode_data = tf.cast(decode_data, tf.uint8)
        encoded_image = tf.image.encode_jpeg(decode_data)
        img_raw = encoded_image.eval()

        if is_save_show_picture:
            local_picture_file = os.path.join(LOCAL_TFRECORD_PATH, image_id, 'resize.jpg')
            with tf.gfile.GFile(local_picture_file, 'wb') as f:
                f.write(img_raw)

        writer = tf.python_io.TFRecordWriter(local_tf)
        example = tf.train.Example(features=tf.train.Features(feature={
                      'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                      'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
                      'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_width])),
                      'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_height])),
                      'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def connect_pai_oss():
    auth = oss2.Auth(OssConfig.access_key_id, OssConfig.access_key_secret)
    bucket = oss2.Bucket(auth, OssConfig.endpoint, OssConfig.bucket)
    return bucket


def prepare_train_data_in_oss():
    bucket = connect_pai_oss()
    raw = [
        (12332, "https://wx1.sinaimg.cn/orj1080/53db7999gy1gd0cxmhljwj215o0rs4qp.jpg", 0),
        (12432, "/Users/liangyue/Documents/frozen_model_vgg_16/2.jpg", 1)
    ]
    for img_id, img_url, label in raw:
        tfrecord_name = f"trains-{img_id}.tfrecord"
        local_tf = os.path.join(LOCAL_TFRECORD_PATH, tfrecord_name)
        oss_tf = os.path.join(OssPath.TF_RECORD_PATH, tfrecord_name)
        image_to_tfrecord(img_url, label, img_id, local_tf)
        bucket.put_object_from_file(oss_tf, local_tf)


if __name__ == "__main__":
    prepare_train_data_in_oss()