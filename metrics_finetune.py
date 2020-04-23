import ssl
import pickle
import urllib.request

import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context


def precess_img(img_file):
    with tf.Session() as sess:
        set_width, set_height = 224, 224
        if img_file.startswith("https:"):
            req = urllib.request.Request(img_file)
            response = urllib.request.urlopen(req)
            img_raw = response.read()
        else:
            img_raw = tf.gfile.FastGFile(img_file, 'rb').read()
        decode_data = tf.image.decode_jpeg(img_raw, channels=3)
        # resize
        decode_data = tf.image.resize_images(decode_data, [set_width, set_height])
        decode_data = tf.cast(decode_data, tf.uint8)
        img = sess.run(decode_data)
    return img


def p():
    test_df_fn = "/Users/liangyue/PycharmProjects/bi_cron_job/backend/works/social_picture/data/test_data.p"
    with open(test_df_fn, "rb") as p:
        df = pickle.load(p)
    count = 0
    img_list = []
    img_labels = []
    for index, row in df.iterrows():
        img_id, img_url, label = row["id"], row["url"], row["label"]
        img = precess_img(img_url)
        img_list.append(img)
        img_labels.append(label)
        count += 1

        if count % 100 == 0:
            print(f"process {count} img")

    print(f"have {count} img in test.")
    test_sample_n = count
    frozen_graph_path = "./outfile/frozen_inference_graph.pb"
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0')
            # logits  classes
            classes = model_graph.get_tensor_by_name('classes:0')

            batch_size = 200
            round_num = int(test_sample_n / batch_size) + 1
            pred = []
            for i in range(round_num):
                sub_data = img_list[i * batch_size: min((i+1)*batch_size, test_sample_n)]
                predicted_label = sess.run(classes, feed_dict={inputs: sub_data})
                print(len(predicted_label), predicted_label.tolist())
                predicted_label = predicted_label.tolist()
                pred += predicted_label
    print(len(img_labels), img_labels)
    print(len(pred), pred)
    accuracy = sum(1 for x, y in zip(img_labels, pred) if x == y) / len(pred)
    print(accuracy)


if __name__ == '__main__':
    p()
