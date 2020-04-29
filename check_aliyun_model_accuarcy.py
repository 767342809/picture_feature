import os
import csv
import pandas as pd

path = "/Users/liangyue/Downloads/aliyun_model_check"
truth_file = "imgTagOut.csv"
pred_file = "test_result_refine.txt"
result_check_file = "prediction_check.csv"


def load_truth_data():
    file_name = os.path.join(path, truth_file)
    df = pd.read_csv(file_name)
    return df


def load_pred_data():
    file_name = os.path.join(path, pred_file)
    df = pd.read_csv(file_name, sep=';', header=None)
    df = df.rename(columns={0: "url", 1: "tags"})
    return df


def find_truth_tags(df, pred_img_url):
    sub_df = df[df["url;标签;领域"].str.contains(pred_img_url)]
    tags = sub_df["url;标签;领域"].values[0].split(";")[1]
    print("truth_tags: ", tags)
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
            print(pred_img_url, pred_tags)

            truth_tags = find_truth_tags(truth_df, pred_img_url)
            print(','.join(truth_tags))
            csv_writer.writerow([pred_img_url, ','.join(truth_tags), ','.join(pred_tags)])
            match_num, truth_num = compute_accuracy(truth_tags, pred_tags)
            all_record_count += truth_num
            all_match_num += match_num
            print(match_num, truth_num)
            if match_num != 0:
                have_match_num += 1

    print(all_img_num, have_match_num, all_match_num, all_record_count)
    print("img_acc: ", have_match_num/all_img_num, " record_acc: ", all_match_num/all_record_count)


if __name__ == "__main__":
    # run()
    df = load_truth_data()
    print(len(df))