# YoloV5使用教程
## 主要步骤
- 配置基本Yolo环境
- 在Roboflow上制作数据集
- 根据预训练好的yolov5/models/yolov5s.yaml和收集的数据集制作.pt模型
- 将.pt格式模型转化为.onnx模型
- 用OnnxLoader.py加载模型并进行预测标注
## 步骤详解
### 配置基本Yolo环境
- 去yolov5官网下下载文件至该文件夹下，anaconda新建空的虚拟环境，进入新的虚拟环境，在yolov5路径下输入命令**pip install -r requirements.txt**安装Yolo依赖，**pip install onnx onnxruntime**安装Onnx依赖
### 在Roboflow上制作数据集
- 使用Roboflow在线标注数据集https://roboflow.com/?ref=ultralytics，导出标注好的数据集到yolov5路径下
### 制作模型
- python train.py --img 640 --batch 16 --epoch 100 --data First/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt **(其中First/data.yaml替换为自己数据集的.yaml配置文件路径，而且打开.yaml文件修改训练集、测试集和验证集路径保证它们能被找到，batchsize别太大，量电脑力而行，别折磨电脑)**
- 训练完的模型储存在yolov5/runs/exp/train/weights中
### 格式转化
- python export.py --weights ./runs/exp/train/weights/best.pt --include onnx engine --img 640转化完成模型格式
### 模型部署预测
- 将.onnx模型拷贝到ONNX文件夹中，待预测的图片放在Data中，执行OnnxLoader.py，如果有特定需求改源码即可

### 尝试merge操作