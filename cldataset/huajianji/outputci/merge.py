import os

# 确保输出文件是空的
with open('ciout.txt', 'w', encoding='utf-8') as ciout, open('cipaiout.txt', 'w', encoding='utf-8') as cipaitout:
    ciout.write('')
    cipaitout.write('')

# 遍历当前目录下的所有文件夹
for folder in os.listdir('.'):
    folder_path = os.path.join('.', folder)
    
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        ci_file_path = os.path.join(folder_path, 'ci.txt')
        cipai_file_path = os.path.join(folder_path, 'cipai.txt')
        
        # 检查文件是否存在
        if os.path.exists(ci_file_path) and os.path.exists(cipai_file_path):
            # 读取并合并文件内容，指定UTF-8编码
            with open(ci_file_path, 'r', encoding='utf-8') as ci, open(cipai_file_path, 'r', encoding='utf-8') as cipait, \
                 open('ciout.txt', 'a', encoding='utf-8') as ciout, open('cipaiout.txt', 'a', encoding='utf-8') as cipaitout:
                ciout.write(f"Folder: {folder}\n")  # 添加文件夹名称
                ciout.write(ci.read())
                ciout.write('\n')  # 添加换行符以分隔不同文件夹的内容
                cipaitout.write(f"Folder: {folder}\n")  # 添加文件夹名称
                cipaitout.write(cipait.read())
                cipaitout.write('\n')  # 添加换行符以分隔不同文件夹的内容
        else:
            print(f"文件 {ci_file_path} 或 {cipai_file_path} 在文件夹 {folder_path} 中不存在。")