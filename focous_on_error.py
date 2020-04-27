import os
import ssl
import urllib.request
import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context


def tf_decode_error(img_file):
    with tf.Session():
        set_width, set_height = 224, 224
        if img_file.startswith("https:"):
            req = urllib.request.Request(img_file)
            response = urllib.request.urlopen(req)
            img_raw = response.read()
        else:
            img_raw = tf.gfile.FastGFile(img_file, 'rb').read()
        decode_data = tf.image.decode_jpeg(img_raw, channels=3)
        decode_data = tf.image.resize_images(decode_data, [set_width, set_height])
        decode_data = tf.cast(decode_data, tf.uint8)
        encoded_image = tf.image.encode_jpeg(decode_data)
        img_raw = encoded_image.eval()

        folder = "./"
        if not os.path.exists(folder):
            os.makedirs(folder)
        local_picture_file = os.path.join(folder, '+resize.jpg')
        if not os.path.exists(local_picture_file):
            with tf.gfile.GFile(local_picture_file, 'wb') as f:
                f.write(img_raw)

        writer = tf.python_io.TFRecordWriter("./trains.tfrecord")
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_width])),
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_height])),
            'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    # 549452_3_37
    # url = "https://asset.yit.com/yit_community/media/202003/327203ee-7b46-4bfb-a8c1-53d88150e442.jpg"
    url = "/Users/liangyue/Documents/frozen_model_vgg_16/riff1.jpg"
    tf_decode_error(url)
