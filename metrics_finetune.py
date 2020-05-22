import os
import csv
import ssl
import pickle
import datetime
import urllib.request

import tensorflow as tf

from Constant import LabelName, TEST_PATH

ssl._create_default_https_context = ssl._create_unverified_context


def precess_img(img_file):
    with tf.Graph().as_default() as img_graph:
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
        with tf.Session(graph=img_graph) as sess1:
            img = sess1.run(decode_data)
    return img


def load_test_data():
    test_df_fn = os.path.join(TEST_PATH, "test_data.p")
    with open(test_df_fn, "rb") as p:
        test_df = pickle.load(p)
    print("test size: ", len(test_df))

    multi_label_file = os.path.join(TEST_PATH, "all_url_multi_label.p")
    with open(multi_label_file, "rb") as p:
        url_multi_label = pickle.load(p)

    grouped = test_df.groupby("url")
    urls = []
    count = 0
    for name, group in grouped:
        urls.append(name)
        count += 1
        if count > 10:
            break
    print(urls)

    new_test_df = url_multi_label[url_multi_label["url"].isin(urls)]
    print(len(new_test_df), new_test_df)
    return new_test_df


def predict_labels(is_multi=False):
    def session_predict():
        with tf.Session(graph=model_graph) as sess:
            if is_multi:
                result = run_multi_label(
                    sess, inputs, outputs, sub_img_data, csv_writer, sub_tags, sub_img_url, len(sub_img_url)
                )
            else:
                result = run_single_label(sess, inputs, outputs, sub_img_data, csv_writer,
                                          sub_tags, sub_img_url, len(sub_img_url))
            return result

    df = load_test_data()
    frozen_graph_path = "./outfilel3b_0.05_1000/frozen_inference_graph.pb"
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    batch_size = 200
    with model_graph.as_default():
        inputs = model_graph.get_tensor_by_name('image_tensor:0')
        # logits  classes
        layer_name = "classes"
        if is_multi:
            layer_name = "multi"
        outputs = model_graph.get_tensor_by_name(layer_name + ':0')
        all_pred = []
        all_label = []
        with open("predict_results.csv", "w", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["url", "predict_label", "predict_label_name", "truth_label_name"]
            )
            sub_img_data, sub_tags, sub_img_url = [], [], []
            count = 0
            for index, row in df.iterrows():
                img_url, tags = row["url"], row["tags"]
                try:
                    img = precess_img(img_url)
                except Exception as e:
                    print(e)
                    print(img_url, tags)
                    continue
                sub_img_data.append(img)
                sub_tags.append(tags)
                sub_img_url.append(img_url)
                count += 1

                if len(sub_img_data) == batch_size:
                    print(f"start process {batch_size} img")
                    sub_pred = session_predict()
                    # for p in sub_pred:
                    #     print("p: ", p)
                    all_pred += sub_pred
                    all_label += sub_tags
                    sub_img_data, sub_tags, sub_img_url = [], [], []
                    break

                if count % batch_size == 0:
                    t = datetime.datetime.today()
                    print(f"{t} -- process {count} picture. {tags}")

            if len(sub_img_data) != 0:
                sub_pred = session_predict()
                all_pred += sub_pred
                all_label += sub_tags
    print(len(all_label), all_label)
    print(len(all_pred), all_pred)
    accuracy = sum(1 for x, y in zip(all_label, all_pred) if len(set(x).intersection(set(y))) > 0) / len(all_pred)
    print("accuracy: ", accuracy)
    accuracy1 = sum(len(set(x).intersection(set(y))) for x, y in zip(all_label, all_pred)) / sum([len(i) for i in all_pred])
    print("accuracy1", accuracy1)


def run_single_label(sess, inputs, outputs, sub_img_data, writer, labels, img_urls, length):
    predicted_label = __predict_single_label(sess, inputs, outputs, sub_img_data)
    # __write_to_file(writer, labels, img_urls, predicted_label, length)
    return predicted_label


def run_multi_label(sess, inputs, outputs, sub_img_data, writer, labels, img_urls, length):
    predicted_label = __predict_multi_label(sess, inputs, outputs, sub_img_data)
    __write_multi_label_to_file(writer, labels, img_urls, predicted_label, length)
    return predicted_label


def __predict_single_label(sess, inputs, outputs, sub_img_data):
    predicted_label = sess.run(outputs, feed_dict={inputs: sub_img_data})
    predicted_label = predicted_label.tolist()
    return predicted_label


def __predict_multi_label(sess, inputs, outputs, sub_img_data):
    predicted_label = sess.run(outputs, feed_dict={inputs: sub_img_data})
    predicted_label = predicted_label.tolist()
    multi_dict = {}
    print("predicted_label len: ", len(predicted_label))
    r = [[] for _ in range(len(sub_img_data))]
    for row, label in predicted_label:
        multi_dict.setdefault(row, []).append(label)
    for k, v in multi_dict.items():
        r[k] = v
    return r


def __write_multi_label_to_file(writer, labels, img_urls, pred_labels, length):
    label_name = LabelName(3)
    for i in range(length):
        temp_pred_labels = pred_labels[i]
        temp_pred_labels_name = ','.join(map(lambda x: label_name[x], temp_pred_labels))
        writer.writerow(
            [
                img_urls[i],
                ','.join(map(str, temp_pred_labels)),
                temp_pred_labels_name,
                labels[i]
            ]
        )


if __name__ == '__main__':
    predict_labels(True)
