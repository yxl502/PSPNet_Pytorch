import os
import urllib.request
import zipfile
import tarfile

# 不存在“data”文件夹时创建
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# 文件夹“weights”不存在时创建
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

# 从这里下载VOC2012数据集。
# 很花时间(约15分钟)
url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

    tar = tarfile.TarFile(target_path)  # 读取tar文件
    tar.extractall(data_dir)  # 解压缩tar
    tar.close()  # 关闭tar文件


