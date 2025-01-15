# 导入必要的库
from collections import Counter

# 读取文件并统计每行的出现次数
def find_duplicates(input_file_path):
    line_counter = Counter()
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行的首尾空格，并将其分割成单词列表
            words = line.strip().split()
            # 将单词列表转换为元组，以便可以被Counter计数
            line_tuple = tuple(words)
            line_counter[line_tuple] += 1
    return line_counter

# 将重复的行写入到output.txt文件中
def write_duplicates(output_file_path, line_counter):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line, count in line_counter.items():
            if count > 1:  # 只写入重复的行
                line_str = ' '.join(line) + '\n'
                file.write(line_str)

# 输入文件路径
input_file_path = 'cipai.txt'  # 将这里的路径替换成你的输入文件路径
# 输出文件路径
output_file_path = 'outcipai.txt'

# 查找重复的行
duplicates = find_duplicates(input_file_path)
# 将重复的行写入到output.txt文件中
write_duplicates(output_file_path, duplicates)

print(f"重复的行已写入到{output_file_path}")