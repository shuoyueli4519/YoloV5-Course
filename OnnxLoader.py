import onnx
import onnxruntime as ort
import numpy as np
import sys
import cv2

CLASSES = ['horse']

class ONNX(object):
    def __init__(self, onnx_path):
        # 检查ONNX模型路径
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print('ONNX model can not find.')
        else:
            print('Model Loaded.')

        # 创建模型会话对象,并设置为开启性能分析
        options = ort.SessionOptions()
        options.enable_profiling = True

        # 加载ONNX模型并创建会话对象
        # self.onnx_session = ort.InferenceSession(onnx_path, options)
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_key_value(self):
        return self.init_image, self.init_image_height, self.init_image_width
    
    def get_input_name(self):
        # 获取输入节点名称
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    
    def get_output_name(self):
        # 获取输出节点名称
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, image_numpy):
        # 获取输入节点的形状
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy
        return input_feed
    
    def inference(self, img_path):
        # 根据ONNX模型推理结果
        img = cv2.imread(img_path)
        self.init_image = img
        self.init_image_width, self.init_image_height, _ = img.shape

        # 图像预处理,(640, 640)是Yolo要求的输入尺寸,
        # resize_img[:, :, ::-1]是将BGR转换为RGB,
        # transpose(2, 0, 1)是将HWC转换为CHW,
        # astype(np.float32)是将数据类型转换为float32,ONNX模型要求的数据类型,
        # /= 255.0是将数据归一化到0-1之间,
        # np.expand_dims(resize_img_transpose, axis=0)是将数据增加一个维度,因为ONNX模型要求的输入数据是4维的,
        # input_feed = self.get_input_feed(resize_img_transpose)是将数据转换为字典形式,方便后续推理,
        # pred = self.onnx_session.run(None, input_feed)[0]是推理结果,是一个列表,列表中的元素是numpy数组,
        resize_img = cv2.resize(img, (640, 640))
        resize_img_transpose = resize_img[:, :, ::-1].transpose(2, 0, 1)
        resize_img_transpose = resize_img_transpose.astype(np.float32)
        resize_img_transpose /= 255.0
        resize_img_transpose = np.expand_dims(resize_img_transpose, axis=0)
        input_feed = self.get_input_feed(resize_img_transpose)
        pred = self.onnx_session.run(None, input_feed)[0]

        return pred, resize_img
    
"""
@brief 非极大值抑制,用于去除重叠的框,只保留置信度最高的框
@param dets: 候选框信息数组,每个候选框包括位置信息和置信度信息,[x1, y1, x2, y2, score, class]
@param thresh: IOU阈值
"""
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    # 计算每个候选框的面积,加一是为了包括边界上的像素点
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    # 按照置信度从小到大排序
    keep = []
    index = scores.argsort()[::-1]

    #寻找具有最高置信度且与其他框不重叠的框
    while index.size > 0:
        i = index[0]
        keep.append(i)
        
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep

# 将中心矩形转化为边角矩形
def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

"""
@brief 对预测框进行过滤,去除置信度低的框,并进行非极大值抑制
@org_box: 原始预测框,包括位置信息和置信度信息,[x, y, w, h, score, class]
@conf_thres: 置信度阈值
@ious_thres: IOU阈值
"""
def filter_box(org_box, conf_thres, ious_thres):
    # 去除纬度为1的维度
    org_box = np.squeeze(org_box)

    # 去除置信度低的框
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf]
    # print('box.shape: ', box.shape)

    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))

    output = []
    for i in range(len(all_cls)):
        current_cls = all_cls[i]
        current_cls_box = []
        current_out_box = []

        for j in range(len(cls)):
            if cls[j] == current_cls:
                box[j][5] = current_cls
                current_cls_box.append(box[j][:6])

        current_cls_box = np.array(current_cls_box)
        current_cls_box = xywh2xyxy(current_cls_box)
        current_out_box = nms(current_cls_box, ious_thres)

        for k in current_out_box:
            output.append(current_cls_box[k])
    output = np.array(output)
    return output

# 绘制预测框
def draw(image, box_data, init_image, width, height):
    print(width, height)
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        x1 = int(x1 * width / 640)
        x2 = int(x2 * width / 640)
        y1 = int(y1 * height / 640)
        y2 = int(y2 * height / 640)
        
        cv2.rectangle(init_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print(x1, y1, x2, y2)
    
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        
    return image, init_image

if __name__ == "__main__":
    # 加载ONNX模型
    onnx_path = './ONNX/best.onnx'
    model = ONNX(onnx_path)

    output, or_img = model.inference('./Data/oppo_test1.jpeg')
    img, img_width, img_height = model.get_key_value()

    outbox = filter_box(output, 0.8, 0.8)
    if len(outbox) == 0:
        print('没有发现物体')
        sys.exit(0)
    else:
        print('总共分有:',outbox.shape[0],'类; 预测框的结构为:',outbox.shape[1],'层.')

    result_image, init_image = draw(or_img, outbox, img, img_width, img_height)
    cv2.imshow('result', result_image)
    cv2.imshow('init', init_image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()