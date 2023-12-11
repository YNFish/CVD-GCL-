import networkx as nx
import re
import pandas as pd
import os

error_list_over = []

def dot_to_gcl(dot_file, outputdir, repr):
    """
    @dot_file: 输入的dot文件
    @outputdir: 输出路径
    """
    # dataset/piccpg/FFmpeg1_0-cpg.dot
    # dataset/Alldot/FFmpeg2_0/1-cpg.dot
    # print(dot_file)
    filename = dot_file.split("/")[-2]  #.split(".")[0]  # FFmpeg1_0
    # print(filename)
    
    if os.path.exists(outputdir+filename):
        print(f"*************************** {outputdir+filename} is existed!")
        # return
    # file_dir = "dataset/node_edge_dataset/"
    node_attr_csv = outputdir+filename+"/"+ f"{repr}-node.csv"
    edge_attr_csv = outputdir+filename+"/"+ f"{repr}-edge.csv"
    
    output_dir = os.path.dirname(node_attr_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} is not exist, Created sucess!")
    
    # print(edge_attr_csv)
    # 解析.dot文件并构图
    try:
        G = nx.drawing.nx_pydot.read_dot(dot_file)
    except:
        print("-------------------------------- error is happened---------------------------")
        return
    
    # 创建空的DataFrame
    dfs = []
    for node, attrs in G.nodes(data=True):
        replaced_attrs = attrs['label'].replace('&lt;', '<').replace('&gt;','>').replace('&amp','&').replace('&quot','"') # 将所有的<>转运字符全部替换
        # print(replaced_attrs)
        # 获取 operator
        left_paren_index = replaced_attrs.find('(') # 第一个左括号
        comma_index = replaced_attrs.find(',') # 第一个逗号
        right_paren_index = replaced_attrs.rfind(')') # 最后一个右括号
        operator = replaced_attrs[left_paren_index+1:comma_index]
        true_code = replaced_attrs[comma_index+1:right_paren_index]
        subscript = re.findall(r'<SUB>(\d+)</SUB>',replaced_attrs)
        if subscript ==[]:
            # os.remove(output_dir)
            error_list_over.append(dot_file)
            return
        else:
            subscript = subscript[0]
        # df = df.append({'node':node, 'operator': operator, 'true_code': true_code, 'subscript': subscript}, ignore_index=True)
        df = pd.DataFrame({'node': [node], 'operator': [operator], 'subscript': [subscript], 'true_code': [''.join(true_code)]})
        dfs.append(df)
    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv(node_attr_csv, index=False)
    
    # 保存节点属性
    # edge_attributes = {}
    dfedge = []
    for edge in G.edges:
        # edge --> ('449', '450', 0) (源结点，目标节点，不知道)
        attributes = G.get_edge_data(*edge)
        edge_attr = attributes['label'].replace('&lt;', '<').replace('&gt;','>').replace('&amp','&').replace('&quot','"') # 将所有的<>转运字符全部替换
        # print(edge_attr)
        edge_repr = edge_attr[1:4]
        left_colon_index = edge_attr.find(':')
        right_quot_index = edge_attr.rfind('"')
        # print(left_colon_index, right_quot_index)
        if left_colon_index+2 == right_quot_index:
            edge_code = ""
        else:
            edge_code = edge_attr[left_colon_index+2: right_quot_index]
            # print(edge_code)
        source_node = edge[0]
        distination_node = edge[1]
        dfe = pd.DataFrame({'source': [source_node], 'distination': [distination_node], 'repr':[edge_repr], 'code':[edge_code]})
        dfedge.append(dfe)
        # edge_attributes[edge] = edge_attr
    result_df_edge = pd.concat(dfedge, ignore_index=True)
    result_df_edge.to_csv(edge_attr_csv, index=False)
        

# 获取子文件夹
def get_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            subfolders.append(subfolder_path)
    return subfolders

 
if __name__ == "__main__":
    # dot_file = "dataset/Alldot/qemu13313_1/1-cpg.dot"
    # outputdir = "dataset/node_edge_dataset/"
    # repr = "cpg"
    # dot_to_gcl(dot_file, outputdir, repr)

    
    dot_dir = "dataset/Alldot"
    outputdir = "dataset/node_edge_dataset/"
    repr = "pdg"
    dot_ext = "-pdg.dot"
    subfolders = get_subfolders(dot_dir)
    # print(subfolders)
    index = 0
    for subfolder in subfolders:
        dot_file = subfolder+"/1"+dot_ext
        print("+++++++++++++++++++++++++++++++++++++++++   ",dot_file)
        dot_to_gcl(dot_file, outputdir, repr)
        index +=1
        print("-----------------------------------------  ")
    print(index,"--------------------end------------------")
   
    f = open('error.txt', 'w') 
    for err_list in error_list_over:
        print(err_list)
        f.write(err_list+'\n')
        
    f.close()