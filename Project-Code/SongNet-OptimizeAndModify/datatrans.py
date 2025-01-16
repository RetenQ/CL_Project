import json


# 读取原始JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# 为段落添加句号
def add_period_to_paragraphs(data):
    for item in data:
        for i, paragraph in enumerate(item['paragraphs']):
            if not paragraph.endswith('。'):
                item['paragraphs'][i] = paragraph + '。'
    return data


# 保存新的JSON文件
def save_json_file(data, new_file_path):
    with open(new_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# 主函数
def main():
    # 假设你的JSON文件名为 original.json
    original_file_path = '/home/data/GYX/llm/datatrans/nalanxingde.json'
    new_file_path = '/home/data/GYX/llm/datatrans/new.json'

    # 读取原始JSON文件
    data = read_json_file(original_file_path)

    # 为段落添加句号
    data = add_period_to_paragraphs(data)

    # 保存新的JSON文件
    save_json_file(data, new_file_path)
    print(f'New JSON file saved as {new_file_path}')


# 运行主函数
if __name__ == '__main__':
    main()