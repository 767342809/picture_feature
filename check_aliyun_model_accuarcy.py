import os
import csv
import pickle
import pandas as pd

from Constant import LabelName

path = "/Users/liangyue/Downloads/aliyun_model_check"
truth_file = "imgTagOut.csv"
pred_file = "test_result_refine.txt"
result_check_file = "prediction_check.csv"
ali_test = "ali_test_data.p"
my_pred_file = "predict_results.csv"
merge_file = "merge_results.csv"


def load_truth_data():
    file_name = os.path.join(path, truth_file)
    df = pd.read_csv(file_name, sep=';')
    return df


def load_pred_data():
    file_name = os.path.join(path, pred_file)
    df = pd.read_csv(file_name, sep=';', header=None)
    df = df.rename(columns={0: "url", 1: "tags"})
    return df


def load_my_pred_data():
    file_name = os.path.join(path, my_pred_file)
    df = pd.read_csv(file_name)
    return df


def find_truth_tags(df, pred_img_url):
    sub_df = df[df["url"].str.contains(pred_img_url)]
    tags = sub_df["标签"].values[0]
    # print("truth_tags: ", tags)
    return tags.split(",")


def compute_accuracy(truth_tag, pred_tag):
    match_num = set(truth_tag).intersection(set(pred_tag))
    return len(match_num), len(truth_tag)


def run():
    truth_df = load_truth_data()
    pred_df = load_pred_data()
    all_img_num = len(pred_df)

    all_record_count = 0
    all_match_num = 0
    have_match_num = 0
    result_file = os.path.join(path, result_check_file)
    with open(result_file, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["url", "truth_tags", "pre_tags"])
        for index, row in pred_df.iterrows():
            pred_img_url = row["url"]
            pred_tags = row["tags"].split(",")
            # print(pred_img_url, pred_tags)

            truth_tags = find_truth_tags(truth_df, pred_img_url)
            # print(','.join(truth_tags))
            csv_writer.writerow([pred_img_url, ','.join(truth_tags), ','.join(pred_tags)])
            match_num, truth_num = compute_accuracy(truth_tags, pred_tags)
            all_record_count += truth_num
            all_match_num += match_num
            # print(match_num, truth_num)
            if match_num != 0:
                have_match_num += 1

            if index % 1000 == 0:
                print(f"process {index} rows. ")

    print(all_img_num, have_match_num, all_match_num, all_record_count)
    print("img_acc: ", have_match_num/all_img_num, " record_acc: ", all_match_num/all_record_count)
    # 6138
    # 3768
    # 4500
    # 9616
    # img_acc: 0.613880742913001
    # record_acc: 0.4679700499168053


def generate_data_and_send_to_file():
    import pickle
    # (new_id, url, label)
    df = load_truth_data()
    all_data = []
    for index, row in df.iterrows():
        url = row["url"]
        tags = row["标签"].split(",")
        r = 0
        for tag in tags:
            new_id = str(index) + "_" + str(r)
            all_data.append((new_id, url, tag))
            r += 1
        if index % 1000 == 0:
            print(f"process {index} img. ")
    with open("all_data_for_finetune.p", "wb") as p:
        pickle.dump(all_data, p)
    print(all_data)


def generate_ali_test():
    ln = LabelName(3)
    truth_df = load_truth_data()
    pred_df = load_pred_data()
    all_img_num = len(pred_df)
    print(all_img_num)

    all_record_count = 0
    test_data = []
    for index, row in pred_df.iterrows():
        pred_img_url = row["url"]
        truth_tags = find_truth_tags(truth_df, pred_img_url)
        try:
            truth_tags_label = list(map(lambda x: ln.index(x), truth_tags))
        except KeyError as _:
            print(pred_img_url, truth_tags)
            continue
        test_data.append((str(all_record_count), pred_img_url, truth_tags_label))
        all_record_count += 1

        if all_record_count % 1000 == 0:
            print(f"process {all_record_count} record")

    df = pd.DataFrame(test_data, columns=["id", "url", "label"])
    print(df)

    file_name = os.path.join(path, ali_test)
    with open(file_name, "wb") as f:
        pickle.dump(df, f)


def img_accuracy(real, pred):
    accuracy = sum(1 for x, y in zip(real, pred) if len(set(x.split(",")).intersection(set(y.split(",")))) > 0) / len(real)
    return accuracy


def record_accuracy(real, pred):
    accuracy = sum([len(set(x.split(",")).intersection(set(y.split(",")))) for x, y in zip(real, pred)]) / sum(
        [len(i.split(",")) for i in real])
    return accuracy


def append_ali_pred():
    ali_df = load_pred_data()
    my_pred_df = load_my_pred_data()
    df1 = pd.merge(my_pred_df, ali_df, on="url")
    df1 = df1.rename(columns={"tags": "ali_pred"})
    df1 = df1[df1["predict_label_name"].notnull()]
    file_name = os.path.join(path, merge_file)
    df1.to_csv(file_name)

    truth = df1["truth_label_name"].to_list()
    ali_pred = df1["ali_pred"].to_list()
    my_pred = df1["predict_label_name"].to_list()

    print("ali img_accuracy: ", img_accuracy(truth, ali_pred), " record_accuracy: ", record_accuracy(truth, ali_pred))
    print("my img_accuracy: ", img_accuracy(truth, my_pred), " record_accuracy: ", record_accuracy(truth, my_pred))


if __name__ == "__main__":
    # run()
    # generate_data_and_send_to_file()
    # generate_ali_test()

    append_ali_pred()