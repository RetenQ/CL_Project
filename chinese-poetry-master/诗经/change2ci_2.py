import re

# 读取output.txt文件
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# 交换作者信息和主题的位置，并保持句子分行
def swap_author_title(line):
    # 使用正则表达式匹配作者信息、主题和内容
    match = re.match(r'(.*)<s1>(.*)<s2>(.*)', line)
    if match:
        author_info, title, content = match.groups()
        # 将内容中的</s>替换回换行符，以便分行
        content_lines = content.split('</s>')
        # 重新添加</s>并格式化字符串
        formatted_content = '</s>'.join([f"{line}</s>" for line in content_lines])
        return f"{title}<s1>{author_info}<s2>{formatted_content}"
    return line

# 将处理后的数据写入output2.txt文件
def write_to_file(lines, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(swap_author_title(line) + '\n')

# 主函数
def main():
    # 输入文件路径
    input_file_path = 'output.txt'
    # 输出文件路径
    output_file_path = 'output2.txt'
    
    # 读取文件数据
    lines = read_file(input_file_path)
    
    # 交换作者信息和主题的位置，并写入新的文件
    write_to_file(lines, output_file_path)

# 执行主函数
if __name__ == '__main__':
    main()