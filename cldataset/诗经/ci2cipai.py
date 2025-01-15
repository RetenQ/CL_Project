import json

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 收集作者信息并按chapter进行分组，同时去除重复项
def collect_and_group_author_info(data):
    author_info_dict = {}
    for element in data:
        author_info = f"{element['chapter']}-{element['section']}"
        if element['chapter'] not in author_info_dict:
            author_info_dict[element['chapter']] = set()  # 使用集合避免重复
        author_info_dict[element['chapter']].add(author_info)
    return author_info_dict

# 将作者信息写入output_cipai.txt文件，按chapter分行
def write_author_info_to_file(author_info_dict, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for chapter, author_infos in author_info_dict.items():
            # 将同一chapter的author_infos以空格分隔写入同一行
            authors = ' '.join(sorted(author_infos))  # 排序确保顺序
            file.write(authors + '\n')

# 主函数
def main():
    # JSON文件路径
    json_file_path = 'shijing.json'
    # 输出文件路径
    output_file_path = 'output_cipai.txt'
    
    # 读取JSON数据
    data = read_json(json_file_path)
    
    # 收集作者信息并按chapter进行分组
    author_info_dict = collect_and_group_author_info(data)
    
    # 写入作者信息到文件，按chapter分行
    write_author_info_to_file(author_info_dict, output_file_path)

# 执行主函数
if __name__ == '__main__':
    main()