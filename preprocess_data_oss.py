import os
import ssl
import tensorflow as tf


class OssPath(object):
    BUCKET_PATH = "oss://yit-prod-pai"
    PREFIX = "ml-pai/image_finetune"
    MODEL_NAME = "resnet_v1_50.ckpt"
    PRE_TRAINED_MODEL_PATH = os.path.join(BUCKET_PATH, PREFIX, "pre_trained_model", MODEL_NAME)
    TRAIN_TF_RECORD_PATH = os.path.join(BUCKET_PATH, PREFIX, "tfrecord_data_test")
    LOG_DIR_PATH = os.path.join(BUCKET_PATH, PREFIX, "training_log")
    INPUT_LIST = os.path.join(BUCKET_PATH, PREFIX, "input.txt")


ssl._create_default_https_context = ssl._create_unverified_context


def image_to_tfrecord(img_file, label, image_id, is_save_show_picture=False):
    with tf.Session():
        set_width, set_height = 224, 224

        img_raw = tf.gfile.FastGFile(img_file, 'rb').read()
        # decode_data = tf.image.decode_jpeg(img_raw, channels=3)
        # decode_data = tf.image.resize_images(decode_data, [set_width, set_height])
        # decode_data = tf.cast(decode_data, tf.uint8)
        # encoded_image = tf.image.encode_jpeg(decode_data)
        # img_raw = encoded_image.eval()


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

        if is_save_show_picture:
            folder = "./resize1"
            if not os.path.exists(folder):
                os.makedirs(folder)
            local_picture_file = os.path.join(folder, image_id + '+resize.jpg')
            if not os.path.exists(local_picture_file):
                with tf.gfile.GFile(local_picture_file, 'wb') as f:
                    f.write(img_raw)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_width])),
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[set_height])),
            'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        return example.SerializeToString()


def save_local(local_tf, tfrecord):
    with tf.Session():
        writer = tf.python_io.TFRecordWriter(local_tf)
        writer.write(tfrecord)
        writer.close()


def prepare_train_data_in_oss():
    files = tf.gfile.FastGFile(OssPath.INPUT_LIST, mode="r")
    count = 0
    for file in files:
        split_f = file.split(",")
        img_path, label = split_f[0], int(split_f[1])
        img_id = img_path.split("/")[-1].split(".")[0]
        tfrecord_name = "trains-"+img_id+".tfrecord"
        oss_tf = os.path.join(OssPath.BUCKET_PATH, OssPath.TRAIN_TF_RECORD_PATH, tfrecord_name)
        if not os.path.exists(oss_tf):
            try:
                tfrecord_img = image_to_tfrecord(img_path, label, img_id, True)
            except Exception as e:
                print("e: ", e)
                print(img_id, img_path)
                continue
            save_local(oss_tf, tfrecord_img)
        else:
            print("exist", oss_tf)

        count += 1
        if count % 100 == 0:
            print("process %d picture." % count)


if __name__ == "__main__":
    prepare_train_data_in_oss()
