import os
import ssl
import oss2
import pickle
import datetime
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
    tf.reset_default_graph()
    writer.close()


def save_local(local_tf, tfrecord):
    with tf.Session():
        writer = tf.python_io.TFRecordWriter(local_tf)
        writer.write(tfrecord)
        writer.close()


def connect_pai_oss():
    auth = oss2.Auth(OssConfig.access_key_id, OssConfig.access_key_secret)
    bucket = oss2.Bucket(auth, OssConfig.endpoint, OssConfig.bucket)
    return bucket


def prepare_train_data_in_oss():
    if not os.path.exists(LOCAL_TFRECORD_PATH):
        os.makedirs(LOCAL_TFRECORD_PATH)
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
        # if not bucket.object_exists(oss_tf):
        try:
            image_to_tfrecord(img_url, label, img_id, local_tf)
        except Exception as e:
            print("e: ", e)
            print(img_id, img_url)
            continue
        bucket.put_object_from_file(oss_tf, local_tf)
        # else:
        #     print("exist", oss_tf)

        count += 1
        if count % 1000 == 0:
            t = datetime.datetime.today()
            print(f"{t} -- process {count} picture. {label}")


def check_tfrecord():
    bucket = connect_pai_oss()

    cnt = 0
    prefix = OssPath.TF_RECORD_PATH+"/t"
    print(prefix)
    # bucket.batch_delete_objects(key_list)
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        bucket.delete_object(obj.key)
        # object_stream = bucket.get_object(obj.key)
        # example = object_stream.read()
        cnt = cnt + 1
        # try:
        #     tf_example = tf.train.Example.FromString(example)
        # except :
        #     # bucket.delete_object(obj.key)
        #     print("removing %s" % obj.key)
        if cnt % 1000 == 0:
            print(f"check {cnt} files. ")


if __name__ == "__main__":
    check_tfrecord()
