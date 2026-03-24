#NEW_FILE_CODE
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
import cv2


class GarbageObjectDetector:
    """垃圾目标检测器 - 使用预训练的目标检测模型"""

    def __init__(self, detection_threshold=0.5, model_type='faster_rcnn'):
        """
        初始化目标检测器

        Args:
            detection_threshold: 检测阈值
            model_type: 模型类型 ('faster_rcnn' 或 'retinanet')
        """
        self.detection_threshold = detection_threshold
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载预训练的目标检测模型
        if model_type == 'faster_rcnn':
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )
        elif model_type == 'retinanet':
            self.model = retinanet_resnet50_fpn(
                weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1
            )
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")

        self.model.to(self.device)
        self.model.eval()

        # COCO 数据集的类别名称（包含一些可回收物类别）
        # 我们主要关注可能与垃圾相关的类别
        self.coco_classes = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
            15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
            21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack',
            26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee',
            31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
            36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
            40: 'bottle',  # 瓶子 - 可能是塑料或玻璃
            41: 'wine glass',  # 酒杯 - 玻璃
            42: 'cup',  # 杯子
            43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl',  # 餐具
            47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange',
            51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut',
            56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed',
            61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse',
            66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven',
            71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock',
            76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'
        }

        # 定义与垃圾分类映射的类别
        self.garbage_relevant_classes = {
            40: 'bottle',  # 瓶子 -> plastic/glass
            41: 'wine glass',  # 酒杯 -> glass
            42: 'cup',  # 杯子 -> plastic/paper
            74: 'book',  # 书 -> paper
            76: 'vase',  # 花瓶 -> glass
        }

        # 为不同垃圾类型定义不同的颜色
        self.class_colors = {
            'cardboard': (255, 165, 0),  # 橙色
            'glass': (0, 255, 255),  # 黄色
            'metal': (192, 192, 192),  # 银色
            'paper': (0, 255, 0),  # 绿色
            'plastic': (0, 0, 255),  # 红色
            'bottle': (255, 0, 255),  # 品红
            'cup': (255, 192, 203),  # 粉色
            'book': (128, 0, 128),  # 紫色
            'default': (0, 255, 0)  # 默认绿色
        }

        # 预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect_objects(self, image_path):
        """
        检测图像中的物体

        Args:
            image_path: 图像路径或 PIL Image 对象

        Returns:
            detections: 检测结果列表，每个结果包含 box, class, confidence
        """
        try:
            # 加载图像
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path

            # 转换为 tensor
            img_tensor = self.transform(image).to(self.device)

            # 进行检测
            with torch.no_grad():
                prediction = self.model([img_tensor])

            # 解析检测结果
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            # 过滤低置信度的检测
            detections = []
            for i in range(len(boxes)):
                if scores[i] >= self.detection_threshold:
                    label_id = int(labels[i])
                    label_name = self.coco_classes.get(label_id, f'class_{label_id}')

                    detection = {
                        'box': boxes[i],  # [x1, y1, x2, y2]
                        'class_id': label_id,
                        'class_name': label_name,
                        'confidence': float(scores[i]),
                        'garbage_type': self._map_to_garbage_type(label_name)
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            print(f"检测失败：{str(e)}")
            return []

    def _map_to_garbage_type(self, class_name):
        """
        将检测到的物体类别映射到垃圾分类类型

        Args:
            class_name: COCO 类别名称

        Returns:
            garbage_type: 垃圾分类类型 (cardboard/glass/metal/paper/plastic)
        """
        mapping = {
            'bottle': 'plastic',  # 瓶子通常是塑料
            'wine glass': 'glass',  # 酒杯是玻璃
            'cup': 'plastic',  # 杯子可能是塑料
            'book': 'paper',  # 书是纸
            'vase': 'glass',  # 花瓶是玻璃
            'fork': 'metal',  # 叉子是金属
            'knife': 'metal',  # 刀是金属
            'spoon': 'metal',  # 勺子是金属
            'bowl': 'ceramic',  # 碗可能是陶瓷
            'scissors': 'metal',  # 剪刀是金属
        }

        return mapping.get(class_name, 'unknown')

    def draw_detections(self, image, detections):
        """
        在图像上绘制检测框

        Args:
            image: PIL Image 或 numpy 数组
            detections: 检测结果列表

        Returns:
            result_image: 绘制了检测框的图像 (numpy 格式)
        """
        # 转换为 OpenCV 格式
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image.copy()

        # 获取图像尺寸用于缩放字体
        height, width = img_cv.shape[:2]
        font_scale = min(width, height) / 640.0
        thickness = max(1, int(2 * font_scale))

        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = map(int, box)

            # 获取颜色
            garbage_type = detection['garbage_type']
            color = self.class_colors.get(garbage_type, self.class_colors['default'])
            color_bgr = (color[2], color[1], color[0])  # RGB -> BGR

            # 绘制边界框
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, thickness + 1)

            # 准备标签文本
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            garbage_label = f"Type: {garbage_type}"

            # 计算文本大小
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 绘制标签背景
            cv2.rectangle(img_cv,
                          (x1, y1 - label_height - 10),
                          (x1 + label_width + 10, y1),
                          color_bgr, -1)

            # 绘制标签文本
            cv2.putText(img_cv, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # 绘制垃圾分类类型
            (type_width, type_height), _ = cv2.getTextSize(
                garbage_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness
            )
            cv2.rectangle(img_cv,
                          (x1, y2 + 5),
                          (x1 + type_width + 10, y2 + type_height + 15),
                          color_bgr, -1)
            cv2.putText(img_cv, garbage_label, (x1 + 5, y2 + type_height + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), thickness)

        return img_cv

    def detect_and_visualize(self, image_path, save_path=None):
        """
        检测并可视化结果

        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）

        Returns:
            result_image: 带检测框的图像
            detections: 检测结果
        """
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')

        # 检测
        detections = self.detect_objects(image_path)

        # 可视化
        result_image = self.draw_detections(image, detections)

        # 保存结果
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"检测结果已保存到：{save_path}")

        return result_image, detections


# 使用示例
if __name__ == '__main__':
    # 创建检测器
    detector = GarbageObjectDetector(detection_threshold=0.5)

    # 检测图像
    image_path = 'test_image.jpg'  # 替换为你的图像路径
    result, detections = detector.detect_and_visualize(image_path, 'result.jpg')

    print(f"检测到 {len(detections)} 个物体:")
    for det in detections:
        print(f"  - {det['class_name']} ({det['garbage_type']}): {det['confidence']:.2%}")
