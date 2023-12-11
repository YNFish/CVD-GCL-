# 1. 目录格式
## 1.1 dataset
1. codebin 保存每个源代码生成的bin文件 code2pic中函数getbin的输出位置
2. dataset_test 测试数据，包括源代码，内部格式是 文件夹名/源代码名
3. node_edge_dataset 经过处理后保存的每个源代码的edge和node属性
    - aaa-bbb-edge.csv
    - aaa-bbb-edge.csv  aaa:源代码名称，bbb:图结构（cpg，ast...）
4. picast code2pic中函数getpic的输出路径，输出的是 ast图的dot
5. piccpg 同上
6. json2code_func.py 将原来的json 格式的数据集变成需要的样式 文件夹/文件名
7. Glove 文件夹,主要完成code embedding
    - dot2vec.py dot文件转化为可以输入到glove的格式, 生成dotGlove.txt
    - vectors.txt 经过 `demo.sh` 生成的关于源代码的embedding 向量表
    - dotvectors.txt 是dot生成的向量表
## 1.2 code2pic.py
用于处理源代码，将源代码转化为需要的 dot文件

## 1.3 dealDot.py
由于joern生成的dot不能直接使用，需要对dot文件进行手动解析

## 1.4 dot2csv.py
通过该文件，将dot文件转化为需要的 csv文件

# 2. 流程
1. 使用 `json2code_func.py` 将 function.json 文件转化为数据集
2. 修改 `code2pic.py` 文件的 输入和输出
3. 使用 `dealDot.py` 对得到的dot 进行处理
4. 使用 `dot2csv.py` 将处理好的dot文件进行解析，输出为csv文件
