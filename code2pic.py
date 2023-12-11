import os
import subprocess

# joern解释器的位置
joern_cli_path = "joern-cli"


# 初始化重试失败列表
retry_failed_list = []

def code2pic(input_folder, repr, output_folder):
    """
    @input_folder: 输入源代码
    @output_folder: 输出的路径
    """
    
    binfile_dir = "dataset/codebin"  # 需要修改 
    # if not os.path.exists(binfile_dir):
    #     os.makedirs(binfile_dir)
    
    # 遍历输入文件夹中的所有源文件
    # for filename in os.listdir(input_folder):
    #     # 使用joern-parse 得到bin文件
    #     getbin_inputdir = "../"+input_folder+"/"+filename
    #     getbin_outputdir = "../dataset/codebin/"
    #     getbin(getbin_inputdir, getbin_outputdir)
    #     print(f"{filename}.bin生成成功，已保存到{getbin_outputdir}中")
    #     os.chdir("..")
    
    print("bin文件全部完成！！！！！！")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建{output_folder}成功")
    for binname in os.listdir(binfile_dir):
        bin2pic_inputdir = "../"+binfile_dir+"/"+binname  # ../dataset/codebin/xxx.bin
        print(bin2pic_inputdir)
        getpic(bin2pic_inputdir,repr, output_folder)
        os.chdir("..")
           
            
def getbin(input_dir, output_dir):
    """
    @input_dir: 输入文件地址
    @output_dir: 输出文件地址
    @joern_command: joern语句类型
    """
    # 进入到joern文件夹
    os.chdir(joern_cli_path)
    
    # 运行joern-parse 命令
    # 保存的文件名 a/b/c 这个文件夹下的c.cpp, 保存的路径是 output_dir+c.bin
    bin_file = output_dir+ os.path.basename(os.path.normpath(input_dir)) + ".bin"
    # 生成.bin 文件
    parse_command = f"./joern-parse {input_dir} -o {bin_file}"
    subprocess.run(parse_command, shell=True)

        

def getpic(input_bin, repr, output_dir):
    """
    @input_bin: 输入文件地址
    @repr: ast,cpg14等
    @output_dir: 输出文件地址
    """
   
    # 进入到joern文件夹
    os.chdir(joern_cli_path)
    # print(os.getcwd())
    
    # 编写export_command命令
    # 获取文件名 a/b/c/d.bin
    filename = os.path.basename(input_bin) # d.bin
    # 提取文件名（去除扩展名）
    name_without_extension = os.path.splitext(filename)[0] # d
    print(name_without_extension)
    outputfile = "../"+output_dir+ name_without_extension # output_dir/d/
    print(outputfile)
    # os.makedirs(outputfile) # 创建新的文件夹
    export_command = f"./joern-export {input_bin} -o {outputfile} --repr={repr}"
    subprocess.run(export_command, shell=True)
    
    
if __name__=='__main__':
    inputdir = "dataset/dataset_all"  # 上一步保存的路径
    output_dir = "dataset/picAll/picAllpdg/" # 保存的路径
    repr = "pdg"  # 要保存的数据格式，可以是 cpg14， ast等，可看joern官网
    print("----------begin----------")
    code2pic(inputdir,repr,output_dir)
    print("----------end------------")
    