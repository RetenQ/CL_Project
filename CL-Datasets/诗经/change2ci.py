import json
import re

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 处理每个元素并格式化为指定字符串
def format_element(element):
    author_info = f"{element['chapter']}-{element['section']}"
    title = element['title']
    sentences = element['content']
    
    # 在每个句子的句号、问号、感叹号后添加</s>
    formatted_sentences = []
    for sentence in sentences:
        # 使用正则表达式找到所有句子结束符号并添加</s>
        formatted_sentence = re.sub(r'([。？！])', r'</s>\1', sentence)
        formatted_sentences.append(formatted_sentence)
    
    # 将所有句子连接成一个字符串
    formatted_content = '</s>'.join(formatted_sentences)
    return f"{author_info}<s1>{title}<s2>{formatted_content}</s>"

# 将处理后的数据写入output.txt文件
def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for element in data:
            formatted_element = format_element(element)
            file.write(formatted_element + '\n')

# 主函数
def main():
    # JSON文件路径
    json_file_path = 'shijing.json'
    # 输出文件路径
    output_file_path = 'output.txt'
    
    # 读取JSON数据
    data = read_json(json_file_path)
    
    # 写入处理后的数据到output.txt
    write_to_file(data, output_file_path)

# 执行主函数
if __name__ == '__main__':
    main()