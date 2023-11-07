import csv

def diff_check():
    with open("データセット_v0.csv", "r") as file1, open("データセット_v0_compare.csv", "r") as file2, open("差分.csv", "w") as file3:
        f1_contents = file1.readlines()
        f2_contents = file2.readlines()

        for (line1, line2) in zip(f1_contents, f2_contents):
            if line1 != line2:
                file3.write(str(line1))
                file3.write(str(line2))

def organize_dataset():
    with open("差分.csv", "r") as diff_file, open("データセット_v0.csv", "r") as dataset_file:
        output = [["id", "app_name", "datatime", "context", "answer"]]
        diff_csv_reader = csv.reader(diff_file)
        diff_rows = list(diff_csv_reader)
        dataset_csv_reader = csv.reader(dataset_file)
        dataset_rows = list(dataset_csv_reader)

        for dataset_row in dataset_rows[1:]:
            for diff_row in diff_rows[1:]:
                if (diff_row[0] == dataset_row[0]) and ("*" in diff_row[4]):
                    checked_answer = diff_row[4].replace("*", "")
                    checked_answer = checked_answer.replace("&", "/")
                    dataset_row[4] = checked_answer
                    break
            output.append(dataset_row)

    with open("データセット_v1.csv", "w", encoding='utf-8', newline='') as dataset_file:
        csv_writer = csv.writer(dataset_file)
        csv_writer.writerows(output)

def main():
    diff_check()
    organize_dataset()

if __name__ == "__main__":
    main()