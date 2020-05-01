import os
import ssl
import oss2
import pickle
import urllib.request

from Constant import OssConfig, OssPath

ssl._create_default_https_context = ssl._create_unverified_context


def read_image_from_url(img_url):
    req = urllib.request.Request(img_url)
    response = urllib.request.urlopen(req)
    img_raw = response.read()
    return img_raw


def connect_pai_oss():
    auth = oss2.Auth(OssConfig.access_key_id, OssConfig.access_key_secret)
    bucket = oss2.Bucket(auth, OssConfig.endpoint, OssConfig.bucket)
    return bucket


def put_img_in_oss():
    bucket = connect_pai_oss()
    train_df_fn = "/Users/liangyue/PycharmProjects/bi_cron_job/backend/works/social_picture/data/train_data.p"
    with open(train_df_fn, "rb") as p:
        df = pickle.load(p)
    classes_count_dict = df.groupby("label").size().reset_index(name='size').to_dict(orient='list')
    print(classes_count_dict)

    with open("input.txt", "a") as t:
        count = 0
        for index, row in df.iterrows():
            img_id, img_url, label = row["id"], row["url"], row["label"]
            file_name = img_id + ".jpg"
            oss_tf = os.path.join(OssPath.IMAGE_PATH, file_name)
            all_oss_path = os.path.join(OssPath.BUCKET_PATH, oss_tf)
            if not bucket.object_exists(oss_tf):
                try:
                    img = read_image_from_url(img_url)
                except Exception as e:
                    print("e: ", e)
                    print(img_id, img_url)
                    continue
                bucket.put_object(oss_tf, img)
                t.write(all_oss_path + "," + str(label) + "\n")
            else:
                print("exist", oss_tf)
            count += 1
            if count % 100 == 0:
                print("input %d image." % count)


if __name__ == "__main__":
    put_img_in_oss()
