"""
@Project :getdataset
@File    :json2code_func.py
@IDE     :PyCharm
@Author  :zbb
@Date    :2023/7/13 22:17
@Action  :本工作主要完成将function.json中的代码提取出来，保存到文件中
"""

import json
import os


def getcode(json_file_path, save_path):
    """
    @json_file_path: 路径
    @:return: null
    """

    project_dict = {}

    # 加载 JSON 数据
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 对于 data 中的每个字典
    for i, dict_item in enumerate(data):
        # 获取项目名和目标
        project = dict_item["project"]
        target = dict_item["target"]

        # 获取代码
        code = dict_item["func"]

        # 对每个项目进行计数
        if project in project_dict:
            project_dict[project] += 1
        else:
            project_dict[project] = 1

        # 保存路径
        print(project_dict[project])

        # 文件名为 "project+target+i.c++", 这样保证每个文件名是独一无二的
        filename = f"{project}{str(project_dict[project])}_{target}.cpp"
        # 放到单独的文件夹
        save_file_path = save_path+ f"{project}{str(project_dict[project])}_{target}/"
        print(save_file_path)
        # 创建新的文件夹
        os.makedirs(save_file_path)
        # 将位置进行拼接
        savepath = save_file_path+filename

        # 写入文件
        with open(savepath, 'w') as f:
            # 替换JSON字符串中的 '\\n' 为文件中实际的换行符 '\n'
            code_formatted = code.replace("\\n", "\n")

            # 按行分隔，过滤掉所有空白行并合并
            filtered_code = "\n".join(line for line in code_formatted.strip().split('\n') if line.strip())

            f.write(filtered_code)


if __name__ == '__main__':
    json_file_path = "dataset/function.json"
    save_path = "dataset/dataset_all/"
    getcode(json_file_path, save_path)
