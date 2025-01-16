import json


# 读取原始JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# 提取内容并保存到TXT文件
def extract_to_txt(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            author = item['author']
            title = item['title']
            paragraphs = item['paragraphs']
            formatted_text = f"{author}<s1>{title}<s2>"
            for i, paragraph in enumerate(paragraphs):
                # 如果不是最后一句，则添加</s>，否则不添加
                if i < len(paragraphs) - 1:
                    formatted_text += f"{paragraph}</s>"
                else:
                    formatted_text += paragraph  # 最后一句不加</s>
            formatted_text += '\n'
            file.write(formatted_text)


# 提取独特标题并按字数分组保存到TXT文件
def extract_unique_titles(data, output_file_path):
    titles_by_length = {}
    for item in data:
        title = item['title']
        length = len(title)
        if length not in titles_by_length:
            titles_by_length[length] = [title]
        else:
            titles_by_length[length].append(title)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for length in sorted(titles_by_length.keys()):
            titles = titles_by_length[length]
            file.write(' '.join(titles) + '\n')


# 主函数
def main():
    # 假设你的JSON文件名为 original.json
    original_file_path = '/home/data/GYX/llm/datatrans/new.json'
    formatted_output_file = '/home/data/GYX/llm/datatrans/ci.txt'
    unique_titles_file = '/home/data/GYX/llm/datatrans/cipai.txt'

    # 读取原始JSON文件
    data = read_json_file(original_file_path)

    # 提取内容并保存到第一个TXT文件
    extract_to_txt(data, formatted_output_file)

    # 提取独特的标题并保存到第二个TXT文件
    extract_unique_titles(data, unique_titles_file)
    print(f'Formatted poems saved to {formatted_output_file}')
    print(f'Unique titles saved to {unique_titles_file}')


# 运行主函数
if __name__ == '__main__':
    main()