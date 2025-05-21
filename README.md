# 金属表面缺陷检测系统

基于YOLOv11与PyQt5的工业质检解决方案，提供缺陷识别与可视化界面

## 📂 项目结构
├── icon/ # 界面图标资源 (2025-05-16)
├── img/ # 示例图像存储 (2025-05-16)
├── output/ # 检测结果输出 (2025-05-21)
├── runs/ # 训练过程日志 (2025-05-16)
├── UI/ # 界面布局文件 (2025-05-16)
├── VOCData/ # VOC格式数据集 (2025-05-16)
├── weights/ # 模型权重文件 (2025-05-16)
│
├── detect.py # 检测推理脚本 (2025-03-22, 1KB)
├── GC10-DET.zip # 原始数据集文件 (2025-05-21, 947MB)
├── GULPY.py # 主界面程序 (2025-05-16, 33KB)
├── requirements.txt # 依赖库列表 (2025-03-22, 1KB)
└── train.py # 模型训练脚本 (2025-03-22, 1KB)


## 🚀 快速开始
### 环境安装
```bash
pip install -r requirements.txt
unzip GC10-DET.zip -d VOCData/  # 解压数据集到指定目录
启动系统
python GULPY.py
🧠 模型训练
# 单GPU训练示例
python train.py --data VOCData/data.yaml --weights '' --batch 16


🌐 系统功能
多输入源支持：本地图片、视频流、实时摄像头
检测结果可视化：置信度热力图、缺陷分类标注
历史记录管理：检测记录存储于output/目录
跨平台运行：支持Windows/Linux/macOS系统


📦 依赖环境
Python 3.8+
PyTorch >= 2.0.0
PyQt5 == 5.15.7
OpenCV-Python >= 4.5.4
📌 注意事项
首次使用需解压GC10-DET.zip到VOCData目录
训练前确保weights目录有预训练模型
实时检测需要连接摄像设备
界面图标资源存放在icon/目录

