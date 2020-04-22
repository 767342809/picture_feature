import os
import ssl
import oss2
import pickle
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

        # resize with padding
        # decode_data = tf.image.resize_image_with_pad(
        #     decode_data, target_height=set_height, target_width=set_width, method=tf.image.ResizeMethod.BILINEAR)
        # decode_data = tf.cast(decode_data, tf.uint8)
        # assert decode_data.shape == (224, 224, 3)
        # encoded_image = tf.image.encode_jpeg(decode_data)
        # print(encoded_image.eval())
        # r = tf.image.decode_jpeg(encoded_image)
        # print(r.shape)
        # img_raw = encoded_image.eval()

        # resize
        decode_data = tf.image.resize_images(decode_data, [set_width, set_height])
        decode_data = tf.cast(decode_data, tf.uint8)
        encoded_image = tf.image.encode_jpeg(decode_data)
        img_raw = encoded_image.eval()

        if is_save_show_picture:
            folder = "./resize"
            if not os.path.exists(folder):
                os.makedirs(folder)
            local_picture_file = os.path.join(folder, image_id+'+resize.jpg')
            if not os.path.exists(local_picture_file):
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
    train_df_fn = "/Users/liangyue/PycharmProjects/bi_cron_job/backend/works/social_picture/data/train_data.p"
    with open(train_df_fn, "rb") as p:
        df = pickle.load(p)
    classes_count_dict = df.groupby("label").size().reset_index(name='size').to_dict(orient='list')
    print(classes_count_dict)

    count = 0
    for index, row in df.iterrows():
        img_id, img_url, label = row["id"], row["url"], row["label"]
        tfrecord_name = f"trains-{img_id}.tfrecord"
        local_tf = os.path.join(LOCAL_TFRECORD_PATH, tfrecord_name)
        oss_tf = os.path.join(OssPath.TF_RECORD_PATH, tfrecord_name)
        try:
            image_to_tfrecord(img_url, label, img_id, local_tf, True)
        except Exception as e:
            print("e: ", e)
            print(img_id, img_url)
            continue

        if not bucket.object_exists(oss_tf):
            bucket.put_object_from_file(oss_tf, local_tf)
        else:
            print("exist", oss_tf)

        count += 1
        if count % 100 == 0:
            print(f"process {count} picture.")


if __name__ == "__main__":
    prepare_train_data_in_oss()
