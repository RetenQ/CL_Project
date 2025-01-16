import json

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 处理每个元素并格式化为指定字符串
def format_element(element):
    author_info = element['author']
    rhythmic = element['rhythmic']
    sentences = element['paragraphs']
    formatted_content = '</s>'.join(sentences)
    return f"{author_info}<s1>{rhythmic}<s2>{formatted_content}"

# 将处理后的数据写入ci.txt文件
def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for element in data:
            formatted_element = format_element(element)
            file.write(formatted_element + '\n')

# 收集主题并写入cipai.txt文件
def collect_and_write_rhythmics(data, file_path):
    rhythmics = set()  # 使用集合来存储唯一的主题
    for element in data:
        rhythmics.add(element['rhythmic'])
    # 将集合转换为列表，并用空格连接每个元素
    rhythmics_list = ' '.join(sorted(rhythmics))
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(rhythmics_list)

# 主函数
def main():
    # JSON文件路径
    json_file_path = 'huajianji-x-juan.json'
    # 输出文件路径
    output_file_path = 'outputci/x/ci.txt'
    output_cipai_path = 'outputci/x/cipai.txt'
    
    # 读取JSON数据
    data = read_json(json_file_path)
    
    # 写入处理后的数据到ci.txt
    write_to_file(data, output_file_path)
    
    # 收集主题并写入cipai.txt
    collect_and_write_rhythmics(data, output_cipai_path)

# 执行主函数
if __name__ == '__main__':
    main()