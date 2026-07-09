# 1. 进入你的数据目录（如果尚未进入）
cd /Users/zhaomingming/data_sets

# 2. 下载数据集（约 7 MB）
curl -L -o coco128.zip https://ultralytics.com/assets/coco128.zip

# 3. 解压
unzip -q coco128.zip -d ./

# 4. 删除 zip 文件（可选）
rm coco128.zip

# 5. 检查目录结构
ls coco128/
# 应该看到 images/ 和 labels/ 两个子目录
