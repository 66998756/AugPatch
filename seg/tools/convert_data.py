import os
import shutil

# 設定基本路徑
base_path = "./data/Foggy_Zurich"
target = "./data/foggy_zurich"
# lists_file_path = os.path.join(base_path, "lists_file_names", "RGB_medium_filenames.txt")
# target_dir = os.path.join(target, "train")
lists_file_path = os.path.join(base_path, "lists_file_names", "RGB_light_filenames.txt")
target_dir = os.path.join(target, "train")
# lists_file_path = os.path.join(base_path, "lists_file_names", "gt_labelTrainIds_testv2_filenames.txt")
# target_dir = os.path.join(target, "gt/val")
# lists_file_path = os.path.join(base_path, "lists_file_names", "RGB_testv2_filenames.txt")
# target_dir = os.path.join(target, "rgb_anon/val")

# 確認目標目錄存在，否則創建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 讀取 lists_file_names 中的 RGB_medium_filenames.txt 檔案
with open(lists_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        image_file = line.strip()
        image_path = os.path.join(base_path, image_file)
        if os.path.exists(image_path):
            dest_path = os.path.join(target_dir, os.path.basename(image_file))
            
            # 移動圖片
            shutil.move(image_path, dest_path)
            print(f"Moved {image_file} to {dest_path}")
        else:
            print(f"File not found: {image_file}")