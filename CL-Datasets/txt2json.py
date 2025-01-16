import json
import os

# 定义函数，将ci.txt文件转换为指定格式并输出为ci.json文件
def convert_txt_to_json(directory):
    # 构造ci.txt文件的完整路径
    txt_file_path = os.path.join(directory, 'ci.txt')

    # 检查ci.txt文件是否存在
    if not os.path.isfile(txt_file_path):
        print(f"Error: {txt_file_path} does not exist.")
        return

    # 读取ci.txt文件
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 创建一个列表，格式化为所需的字典形式
    data = [{"text": line.strip()} for line in lines]

    # 输出为ci.json文件
    json_file_path = os.path.join(directory, 'ci.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f"Conversion completed. The output file is {json_file_path}")

# 调用函数并传入目录路径
directory = "huajianji"  # 替换为你的实际目录路径
convert_txt_to_json(directory)
