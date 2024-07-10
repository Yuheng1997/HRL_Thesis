import csv


def merge_tsv_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r', newline='') as file1, \
            open(file2_path, 'r', newline='') as file2, \
            open(output_path, 'w', newline='') as output_file:

        reader1 = csv.reader(file1, delimiter='\t')
        reader2 = csv.reader(file2, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        # 写入file1的数据
        for row in reader1:
            writer.writerow(row)

        # 写入file2的数据
        for row in reader2:
            writer.writerow(row)


if __name__ == '__main__':
    # 使用函数合并两个TSV文件
    merge_tsv_files('data.tsv', 'violate_data_1_2.tsv', 'merged_file.tsv')