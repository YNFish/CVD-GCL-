import os
import shutil

frontdir = "dataset/node_edge_dataset/"

with open('error.txt','r') as f:
    contents = f.readlines()
f.close()

index = 0
for content in contents:
    print(content)
    backdir = content.split('/')[2]
    alldir = frontdir + backdir
    if os.path.exists(alldir):
        shutil.rmtree(alldir)
    