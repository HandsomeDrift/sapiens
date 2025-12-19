import os

# 需要清理的文件夹路径
folder_path = "/home/xiangxiantong/sapiens/pose/demo/data/itw_videos/mytest/2r_right"

# 遍历文件夹
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # 判断是文件且不以"_img.png"结尾
    if os.path.isfile(file_path) and not filename.endswith("_img.png"):
        try:
            os.remove(file_path)
            print(f"已删除: {file_path}")
        except Exception as e:
            print(f"删除失败: {file_path}, 错误: {e}")
