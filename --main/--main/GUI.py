import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QWidget, QTextEdit,
                             QFrame, QMessageBox, QProgressBar, QListWidget, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QLinearGradient, QColor, QIcon
import json
from PIL import Image
import traceback
import time
import torchvision.transforms as transforms
# 在第 18 行后面添加
from object_detector import GarbageObjectDetector
import cv2


# 使用与训练时完全一致的模型定义
class EfficientGarbageClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientGarbageClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)

        # 修改最后的全连接层（与训练时完全一致）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# 分类线程
class ClassificationThread(QThread):
    finished = pyqtSignal(str, float, QPixmap, str)  # 增加文件路径
    error = pyqtSignal(str)

    def __init__(self, image_path, model, classes):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.classes = classes

    def run(self):
        try:
            print(f"开始处理图像: {self.image_path}")

            # 1. 加载图像
            try:
                # 使用PIL加载图像，确保与训练时一致
                pil_image = Image.open(self.image_path).convert('RGB')
                print(f"原始图像尺寸: {pil_image.size}, 模式: {pil_image.mode}")

            except Exception as e:
                self.error.emit(f"图像加载失败: {str(e)}")
                return

            # 2. 图像预处理 - 与训练时完全一致！
            try:
                # 使用与训练时完全相同的预处理流程
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

                img_tensor = transform(pil_image).unsqueeze(0)

                print(f"处理后的张量形状: {img_tensor.shape}")
                print(f"张量数值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                print(f"张量均值: {img_tensor.mean().item():.3f}, 标准差: {img_tensor.std().item():.3f}")

            except Exception as e:
                self.error.emit(f"图像预处理失败: {str(e)}")
                return

            # 3. 模型预测
            try:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    print(f"模型原始输出: {outputs}")

                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted = torch.max(probabilities, 0)

                class_name = self.classes[predicted.item()]
                confidence_value = confidence.item()

                print(f"预测结果: {class_name}, 置信度: {confidence_value:.4f}")

                # 输出所有类别的概率
                for i, prob in enumerate(probabilities):
                    print(f"  {self.classes[i]}: {prob.item():.4f}")

            except Exception as e:
                self.error.emit(f"模型预测失败: {str(e)}")
                return

            # 4. 准备结果显示
            try:
                # 使用PIL图像进行显示，确保一致性
                display_img = pil_image.copy()
                display_img = display_img.resize((500, 500), Image.Resampling.LANCZOS)

                # 转换为QPixmap
                display_img = display_img.convert("RGB")
                data = display_img.tobytes("raw", "RGB")
                q_img = QImage(data, display_img.size[0], display_img.size[1], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                self.finished.emit(class_name, confidence_value, pixmap, self.image_path)

            except Exception as e:
                self.error.emit(f"结果显示准备失败: {str(e)}")
                return

        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = f"分类过程出错: {str(e)}"
            print(f"详细错误: {error_traceback}")
            self.error.emit(error_msg)


# 添加新的 ObjectDetectionThread 类
class ObjectDetectionThread(QThread):
    finished = pyqtSignal(QPixmap, list)
    error = pyqtSignal(str)

    def __init__(self, image_path, detection_threshold=0.5):
        super().__init__()
        self.image_path = image_path
        self.detection_threshold = detection_threshold

    def run(self):
        try:
            detector = GarbageObjectDetector(detection_threshold=self.detection_threshold)
            result_image, detections = detector.detect_and_visualize(self.image_path)

            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_image_rgb)
            result_pil = result_pil.resize((500, 500), Image.Resampling.LANCZOS)

            data = result_pil.tobytes("raw", "RGB")
            q_img = QImage(data, result_pil.size[0], result_pil.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            self.finished.emit(pixmap, detections)
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = f"目标检测出错：{str(e)}"
            print(f"详细错误：{error_traceback}")
            self.error.emit(error_msg)


class GradientWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(74, 163, 80))
        gradient.setColorAt(0.5, QColor(86, 188, 92))
        gradient.setColorAt(1.0, QColor(74, 163, 80))
        painter.fillRect(self.rect(), gradient)


class HistoryManager:
    """历史记录管理器"""

    def __init__(self, max_history=50):
        self.history_file = "classification_history.json"
        self.max_history = max_history
        self.history = self.load_history()

    def load_history(self):
        """加载历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return []

    def save_history(self):
        """保存历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history[-self.max_history:], f, indent=2, ensure_ascii=False)
        except:
            pass

    def add_record(self, image_path, class_name, confidence, timestamp):
        """添加记录"""
        record = {
            'image_path': image_path,
            'class_name': class_name,
            'confidence': confidence,
            'timestamp': timestamp,
            'filename': os.path.basename(image_path)
        }
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self.save_history()

    def get_recent_history(self, count=10):
        """获取最近记录"""
        return self.history[-count:][::-1]


class GarbageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
        self.current_image_path = None
        self.original_pixmap = None
        self.history_manager = HistoryManager()
        self.batch_mode = False
        self.batch_files = []

        self.detection_mode = False  # 新增
        self.detection_thread = None  # 新增

        sys.excepthook = self.global_exception_handler
        self.init_ui()
        self.load_model()

    def global_exception_handler(self, exctype, value, traceback):
        print(f"全局异常: {exctype}, {value}")
        QMessageBox.critical(None, "意外错误", f"程序遇到意外错误:\n\n{value}\n\n程序将尝试继续运行。")

    def init_ui(self):
        self.setWindowTitle("垃圾分类检测系统")
        self.setFixedSize(1600, 1000)

        self.setWindowIcon(self.create_recycle_icon())


        # 现代化样式表
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 #f8fff8, stop: 0.5 #e8f5e8, stop: 1 #d8eed8);
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #4CAF50, stop: 0.8 #388E3C, stop: 1 #2E7D32);
                border: none;
                color: white;
                padding: 12px 25px;
                font-size: 14px;
                border-radius: 10px;
                min-width: 120px;
                min-height: 40px;
                font-weight: bold;
                margin: 3px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #66BB6A, stop: 0.8 #4CAF50, stop: 1 #388E3C);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #2E7D32, stop: 0.8 #1B5E20, stop: 1 #0D4013);
            }
            QPushButton:disabled {
                background: #C8E6C9;
                color: #81C784;
            }
            QListWidget {
                border: 2px solid #4CAF50;
                border-radius: 10px;
                background-color: white;
                font-size: 12px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #E8F5E9;
            }
            QListWidget::item:selected {
                background-color: #C8E6C9;
                color: #1B5E20;
            }
        """)

        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 15, 20, 15)

        # 左侧主内容区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # 标题栏
        title_widget = GradientWidget()
        title_layout = QVBoxLayout(title_widget)
        title_label = QLabel("♻️ 垃圾分类检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: white;
                padding: 10px;
                background: transparent;
            }
        """)
        title_layout.addWidget(title_label)
        left_layout.addWidget(title_widget)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # 图像显示区域
        image_frame = QFrame()
        image_frame.setFixedWidth(600)
        image_layout = QVBoxLayout(image_frame)

        image_title = QLabel("📷 图像预览")
        image_title.setStyleSheet("""
            font-size: 16px; 
            color: #2E7D32; 
            font-weight: bold; 
            padding: 5px 0px;
            margin: 0px;
        """)
        image_title.setAlignment(Qt.AlignCenter)
        image_title.setFixedHeight(25)
        image_layout.addWidget(image_title)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #F8FDF8;
                border: 3px dashed #A5D6A7;
                border-radius: 15px;
                min-height: 500px;
                font-size: 14px;
                color: #81C784;
                padding: 15px;
            }
        """)
        self.image_label.setText("📁 请选择图片进行检测")
        self.image_label.setMinimumHeight(500)
        image_layout.addWidget(self.image_label)

        # 按钮区域
        button_layout = QHBoxLayout()
        self.select_btn = QPushButton("📂 选择单张图片")
        self.select_btn.clicked.connect(self.select_image)

        self.batch_btn = QPushButton("📁 选择批量图片")
        self.batch_btn.clicked.connect(self.select_batch_images)

        self.detect_btn = QPushButton("🔍 开始检测")
        self.detect_btn.clicked.connect(self.detect_image)
        self.detect_btn.setEnabled(False)

        # 新增目标检测模式按钮
        self.object_detect_btn = QPushButton("🎯 目标检测模式")
        self.object_detect_btn.setCheckable(True)
        self.object_detect_btn.clicked.connect(self.toggle_detection_mode)

        self.debug_btn = QPushButton("🐛 调试模式")
        self.debug_btn.clicked.connect(self.toggle_debug_mode)

        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.batch_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addWidget(self.debug_btn)
        image_layout.addLayout(button_layout)

        # 结果显示区域
        result_frame = QFrame()
        result_frame.setFixedWidth(600)
        result_layout = QVBoxLayout(result_frame)

        result_title = QLabel("📊 检测结果")
        result_title.setStyleSheet("""
            font-size: 16px; 
            color: #2E7D32; 
            font-weight: bold; 
            padding: 5px 0px;
            margin: 0px;
        """)
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setFixedHeight(25)
        result_layout.addWidget(result_title)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(450)
        self.result_text.setPlaceholderText("检测结果将显示在这里...")
        self.result_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 20px;
                font-size: 18px;
                background-color: white;
                color: #1B5E20;
                font-weight: 500;
                line-height: 1.5;
            }
        """)
        result_layout.addWidget(self.result_text)

        # 置信度区域
        confidence_layout = QVBoxLayout()
        confidence_label = QLabel("📈 置信度")
        confidence_label.setStyleSheet("font-size: 18px; color: #2E7D32; font-weight: bold; margin-top: 10px;")
        confidence_label.setAlignment(Qt.AlignCenter)
        confidence_layout.addWidget(confidence_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("置信度: %p%")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                text-align: center;
                height: 30px;
                font-size: 16px;
                font-weight: bold;
                color: #1B5E20;
                background-color: #F1F8E9;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                        stop: 0 #4CAF50, stop: 0.5 #66BB6A, stop: 1 #81C784);
                border-radius: 6px;
            }
        """)
        confidence_layout.addWidget(self.confidence_bar)
        result_layout.addLayout(confidence_layout)

        # 系统状态
        self.status_label = QLabel("✅ 系统状态: 就绪")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #1B5E20;
                background-color: #E8F5E9;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #A5D6A7;
                font-weight: 500;
                margin-top: 10px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.status_label)

        content_layout.addWidget(image_frame)
        content_layout.addWidget(result_frame)
        left_layout.addLayout(content_layout)

        # 右侧历史记录区域
        right_widget = QWidget()
        right_widget.setFixedWidth(300)
        right_layout = QVBoxLayout(right_widget)

        history_title = QLabel("📋 检测历史")
        history_title.setStyleSheet("font-size: 16px; color: #2E7D32; font-weight: bold; margin-bottom: 5px;")
        history_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(history_title)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        right_layout.addWidget(self.history_list)

        # 历史记录操作按钮
        history_btn_layout = QHBoxLayout()
        self.clear_history_btn = QPushButton("清空历史")
        self.clear_history_btn.clicked.connect(self.clear_history)
        self.refresh_history_btn = QPushButton("刷新")
        self.refresh_history_btn.clicked.connect(self.refresh_history)

        history_btn_layout.addWidget(self.clear_history_btn)
        history_btn_layout.addWidget(self.refresh_history_btn)
        right_layout.addLayout(history_btn_layout)

        # 添加到主布局
        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 1)

        # 底部状态栏
        self.statusBar().setStyleSheet("""
            QStatusBar {
                font-size: 12px;
                font-weight: 500; 
                color: #2E7D32; 
                background-color: #E8F5E9;
                border-top: 1px solid #C8E6C9;
                padding: 4px;
            }
        """)
        self.statusBar().showMessage("✅ 系统就绪 - 请选择图片进行检测")

        # 初始化历史记录
        self.refresh_history()

        # 调试模式
        self.debug_mode = False

    def create_recycle_icon(self):
        """创建回收符号图标"""
        pixmap = QPixmap(48, 48)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("Segoe UI Emoji", 32)
        painter.setFont(font)
        painter.setPen(QColor(76, 175, 80))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "♻️")
        painter.end()

        return QIcon(pixmap)

    def load_model(self):
        """安全加载模型"""
        try:
            self.status_label.setText("🔄 系统状态: 正在加载模型...")
            self.statusBar().showMessage("正在加载模型...")

            # 使用与训练时完全相同的模型结构
            self.model = EfficientGarbageClassifier(num_classes=5)

            model_path = "models/best_model.pth"
            if not os.path.exists(model_path):
                self.show_error(f"找不到模型文件: {model_path}")
                return

            print(f"尝试加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')

            # 详细检查模型权重
            print("检查点包含的键:", checkpoint.keys())

            # 直接加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果直接是模型状态字典
                self.model.load_state_dict(checkpoint)

            self.model.eval()

            val_acc = checkpoint.get('val_acc', 0)
            train_acc = checkpoint.get('train_acc', 0)

            self.status_label.setText("✅ 系统状态: 模型加载成功")
            self.statusBar().showMessage(f"✅ 模型加载成功 - 验证准确率: {val_acc:.2%}")

            print(f"模型加载成功，验证准确率: {val_acc:.2%}, 训练准确率: {train_acc:.2%}")

            # 测试模型推理
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                test_output = self.model(test_input)
                print(f"测试推理成功，输出形状: {test_output.shape}")

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            print(f"模型加载错误: {traceback.format_exc()}")
            self.show_error(error_msg)

    def toggle_debug_mode(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.debug_btn.setStyleSheet("background: #FF9800; color: white;")
            self.debug_btn.setText("🐛 调试模式: 开启")
            self.statusBar().showMessage("🔧 调试模式已开启")
        else:
            self.debug_btn.setStyleSheet("")
            self.debug_btn.setText("🐛 调试模式")
            self.statusBar().showMessage("✅ 调试模式已关闭")

    def select_image(self):
        """选择单张图片文件"""
        self.batch_mode = False
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)"
            )

            if file_path:
                self.current_image_path = file_path
                self.load_and_display_image(file_path)

        except Exception as e:
            self.show_error(f"选择图片时出错: {str(e)}")

    def select_batch_images(self):
        """选择批量图片文件"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择批量图片", "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)"
            )

            if file_paths:
                self.batch_mode = True
                self.batch_files = file_paths
                self.current_image_path = file_paths[0]
                self.load_and_display_image(file_paths[0])
                self.status_label.setText(f"📁 系统状态: 已选择 {len(file_paths)} 张图片")
                self.statusBar().showMessage(f"已选择 {len(file_paths)} 张图片 - 点击开始检测")

        except Exception as e:
            self.show_error(f"选择批量图片时出错: {str(e)}")

    def load_and_display_image(self, file_path):
        """加载并显示图片"""
        if not os.path.exists(file_path):
            self.show_error("文件不存在")
            return

        try:
            # 使用PIL加载图像以确保一致性
            pil_image = Image.open(file_path).convert('RGB')

            # 转换为QPixmap用于显示
            pil_image_display = pil_image.resize((550, 500), Image.Resampling.LANCZOS)
            pil_image_display = pil_image_display.convert("RGB")
            data = pil_image_display.tobytes("raw", "RGB")
            q_img = QImage(data, pil_image_display.size[0], pil_image_display.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            self.original_pixmap = pixmap.copy()
            self.image_label.setPixmap(pixmap)

            self.detect_btn.setEnabled(True)
            self.result_text.clear()
            self.confidence_bar.setValue(0)

            if not self.batch_mode:
                self.status_label.setText(f"📷 系统状态: 图片已选择 - {os.path.basename(file_path)}")
                self.statusBar().showMessage(f"✅ 图片已选择 - 点击开始检测")

        except Exception as e:
            self.show_error(f"图片预览失败: {str(e)}")

    def toggle_detection_mode(self):
        """切换目标检测模式"""
        self.detection_mode = self.object_detect_btn.isChecked()
        if self.detection_mode:
            self.object_detect_btn.setStyleSheet("background: #FF9800; color: white;")
            self.object_detect_btn.setText("🎯 目标检测：开启")
            self.statusBar().showMessage("🎯 目标检测模式已开启 - 将识别多个物体")
            self.detect_btn.setText("🎯 开始目标检测")
        else:
            self.object_detect_btn.setStyleSheet("")
            self.object_detect_btn.setText("🎯 目标检测模式")
            self.statusBar().showMessage("✅ 目标检测模式已关闭")
            self.detect_btn.setText("🔍 开始分类检测")

    def detect_image(self):
        """开始检测图片"""
        if not self.current_image_path or not self.model:
            self.show_error("请先选择图片并确保模型已加载")
            return

        if self.detection_mode:
            self.process_object_detection()
        elif self.batch_mode and self.batch_files:
            self.process_batch_images()
        else:
            self.process_single_image()

    def process_single_image(self):
        """处理单张图片"""
        if not os.path.exists(self.current_image_path):
            self.show_error("图片文件不存在，请重新选择")
            return

        try:
            self.detect_btn.setEnabled(False)
            self.select_btn.setEnabled(False)
            self.batch_btn.setEnabled(False)
            self.debug_btn.setEnabled(False)
            self.status_label.setText("🔍 系统状态: 正在检测...")
            self.statusBar().showMessage("正在检测图片...")

            self.classification_thread = ClassificationThread(
                self.current_image_path, self.model, self.classes
            )
            self.classification_thread.finished.connect(self.on_classification_finished)
            self.classification_thread.error.connect(self.on_classification_error)
            self.classification_thread.start()

        except Exception as e:
            self.show_error(f"开始检测时出错: {str(e)}")
            self.enable_buttons()

    def process_object_detection(self):
        """处理目标检测"""
        if not os.path.exists(self.current_image_path):
            self.show_error("图片文件不存在，请重新选择")
            return

        try:
            self.detect_btn.setEnabled(False)
            self.select_btn.setEnabled(False)
            self.batch_btn.setEnabled(False)
            self.debug_btn.setEnabled(False)
            self.object_detect_btn.setEnabled(False)
            self.status_label.setText("🎯 系统状态：正在检测目标...")
            self.statusBar().showMessage("正在检测图像中的物体...")

            self.detection_thread = ObjectDetectionThread(
                self.current_image_path, detection_threshold=0.5
            )
            self.detection_thread.finished.connect(self.on_object_detection_finished)
            self.detection_thread.error.connect(self.on_classification_error)
            self.detection_thread.start()

        except Exception as e:
            self.show_error(f"开始检测时出错：{str(e)}")
            self.enable_buttons()

    def process_batch_images(self):
        """处理批量图片"""
        if not self.batch_files:
            return

        self.batch_results = []
        self.current_batch_index = 0
        self.process_next_batch_image()

    def process_next_batch_image(self):
        """处理下一张批量图片"""
        if self.current_batch_index >= len(self.batch_files):
            # 批量处理完成
            self.show_batch_results()
            return

        current_file = self.batch_files[self.current_batch_index]
        self.current_image_path = current_file
        self.load_and_display_image(current_file)

        try:
            self.status_label.setText(f"🔍 系统状态: 正在检测 ({self.current_batch_index + 1}/{len(self.batch_files)})")
            self.statusBar().showMessage(f"批量检测中: {self.current_batch_index + 1}/{len(self.batch_files)}")

            self.classification_thread = ClassificationThread(
                current_file, self.model, self.classes
            )
            self.classification_thread.finished.connect(self.on_batch_classification_finished)
            self.classification_thread.error.connect(self.on_batch_classification_error)
            self.classification_thread.start()

        except Exception as e:
            print(f"批量检测出错: {str(e)}")
            self.current_batch_index += 1
            QTimer.singleShot(100, self.process_next_batch_image)

    def on_batch_classification_finished(self, class_name, confidence, pixmap, image_path):
        """批量分类完成回调"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.history_manager.add_record(image_path, class_name, confidence, timestamp)

        self.batch_results.append({
            'file': image_path,
            'class': class_name,
            'confidence': confidence,
            'timestamp': timestamp
        })

        self.current_batch_index += 1
        QTimer.singleShot(100, self.process_next_batch_image)

    def on_batch_classification_error(self, error_msg):
        """批量分类错误回调"""
        print(f"批量检测错误: {error_msg}")
        self.current_batch_index += 1
        QTimer.singleShot(100, self.process_next_batch_image)

    def show_batch_results(self):
        """显示批量结果"""
        if not self.batch_results:
            return

        # 统计结果
        class_count = {}
        total_confidence = 0

        for result in self.batch_results:
            class_name = result['class']
            class_count[class_name] = class_count.get(class_name, 0) + 1
            total_confidence += result['confidence']

        avg_confidence = total_confidence / len(self.batch_results)

        # 生成报告
        report = f"🎉 批量检测完成!\n\n"
        report += f"📊 总体统计:\n"
        report += f"   总图片数: {len(self.batch_results)}\n"
        report += f"   平均置信度: {avg_confidence:.1%}\n\n"
        report += f"📈 分类分布:\n"

        for class_name, count in class_count.items():
            percentage = (count / len(self.batch_results)) * 100
            icon_dict = {'cardboard': '📦', 'glass': '🥛', 'metal': '🥫', 'paper': '📄', 'plastic': '🧴'}
            icon = icon_dict.get(class_name, '🗑️')
            report += f"   {icon} {class_name}: {count}张 ({percentage:.1f}%)\n"

        self.result_text.setText(report)
        self.confidence_bar.setValue(int(avg_confidence * 100))
        self.enable_buttons()
        self.status_label.setText("✅ 系统状态: 批量检测完成")
        self.statusBar().showMessage(f"✅ 批量检测完成 - 共处理 {len(self.batch_results)} 张图片")
        self.refresh_history()

    def on_classification_finished(self, class_name, confidence, pixmap, image_path):
        """单张分类完成回调"""
        try:
            confidence_percent = confidence * 100

            # 添加历史记录
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.history_manager.add_record(image_path, class_name, confidence, timestamp)

            # 显示结果
            icon_dict = {
                'cardboard': '📦',
                'glass': '🥛',
                'metal': '🥫',
                'paper': '📄',
                'plastic': '🧴',
            }
            icon = icon_dict.get(class_name, '🗑️')

            # 显示详细结果
            result_text = f"""🎉 检测完成!

{icon} 垃圾类别: {class_name.upper()}

📊 置信度: {confidence_percent:.1f}%

⏰ 检测时间: {timestamp}

📁 文件: {os.path.basename(image_path)}"""

            # 如果是调试模式，显示更多信息
            if self.debug_mode:
                result_text += f"\n\n🔧 调试信息:\n   图像路径: {image_path}\n   原始置信度: {confidence:.4f}"

            self.result_text.setText(result_text)
            self.confidence_bar.setValue(int(confidence_percent))

            # 更新图像显示
            self.image_label.setPixmap(pixmap)

            self.enable_buttons()
            self.status_label.setText(f"✅ 系统状态: 检测完成 - {class_name}")
            self.statusBar().showMessage(f"✅ 检测完成: {class_name} (置信度: {confidence_percent:.1f}%)")

            # 刷新历史记录
            self.refresh_history()

        except Exception as e:
            self.show_error(f"结果显示时出错: {str(e)}")
            self.enable_buttons()

    def on_object_detection_finished(self, pixmap, detections):
        """目标检测完成回调"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # 显示结果
            if len(detections) == 0:
                result_text = f"⚠️ 未检测到任何物体\n\n⏰ 检测时间：{timestamp}"
            else:
                result_text = f"🎉 检测完成!\n\n"
                result_text += f"📊 检测到 {len(detections)} 个物体:\n\n"

                # 统计垃圾分类类型
                garbage_stats = {}
                for i, det in enumerate(detections, 1):
                    garbage_type = det['garbage_type']
                    if garbage_type != 'unknown':
                        garbage_stats[garbage_type] = garbage_stats.get(garbage_type, 0) + 1

                    # 添加历史记录
                    self.history_manager.add_record(
                        self.current_image_path,
                        f"{det['class_name']}({garbage_type})",
                        det['confidence'],
                        timestamp
                    )

                    # 添加详细信息
                    icon_dict = {
                        'cardboard': '📦', 'glass': '🥛', 'metal': '🥫',
                        'paper': '📄', 'plastic': '🧴', 'unknown': '❓'
                    }
                    icon = icon_dict.get(garbage_type, '❓')
                    result_text += f"{i}. {icon} {det['class_name']}\n"
                    result_text += f"   垃圾分类：{garbage_type.upper()}\n"
                    result_text += f"   置信度：{det['confidence']:.1%}\n"
                    result_text += f"   位置：[{det['box'][0]:.0f}, {det['box'][1]:.0f}, {det['box'][2]:.0f}, {det['box'][3]:.0f}]\n\n"

                if garbage_stats:
                    result_text += f"\n📈 垃圾分类统计:\n"
                    for gtype, count in garbage_stats.items():
                        icon = icon_dict.get(gtype, '❓')
                        result_text += f"   {icon} {gtype}: {count}个\n"

                # 如果是调试模式，显示更多信息
                if self.debug_mode:
                    result_text += f"\n🔧 调试信息:\n"
                    result_text += f"   图像路径：{self.current_image_path}\n"
                    result_text += f"   检测阈值：0.5\n"
                    result_text += f"   检测算法：Faster R-CNN (COCO)\n"

                result_text += f"\n⏰ 检测时间：{timestamp}"

                self.result_text.setText(result_text)

                # 计算平均置信度
                if detections:
                    avg_confidence = sum(det['confidence'] for det in detections) / len(detections)
                    self.confidence_bar.setValue(int(avg_confidence * 100))
                else:
                    self.confidence_bar.setValue(0)

                # 更新图像显示（带检测框）
                self.image_label.setPixmap(pixmap)

                self.enable_buttons()
                self.status_label.setText(f"✅ 系统状态：检测完成 - 发现 {len(detections)} 个物体")
                self.statusBar().showMessage(f"✅ 检测完成：共 {len(detections)} 个物体")

                # 刷新历史记录
                self.refresh_history()

        except Exception as e:
            self.show_error(f"结果显示时出错：{str(e)}")
            self.enable_buttons()

    def enable_buttons(self):
        """启用所有按钮"""
        self.detect_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.debug_btn.setEnabled(True)
        self.object_detect_btn.setEnabled(True)  # 新增

    def on_classification_error(self, error_msg):
        """分类错误回调"""
        self.show_error(error_msg)
        self.enable_buttons()
        self.status_label.setText("❌ 系统状态: 检测失败")
        self.statusBar().showMessage("❌ 检测失败")

    def refresh_history(self):
        """刷新历史记录"""
        self.history_list.clear()
        recent_history = self.history_manager.get_recent_history(20)

        for record in recent_history:
            icon_dict = {'cardboard': '📦', 'glass': '🥛', 'metal': '🥫', 'paper': '📄', 'plastic': '🧴'}
            icon = icon_dict.get(record['class_name'], '🗑️')
            item_text = f"{icon} {record['class_name']} ({record['confidence']:.1%})\n{record['filename']}\n{record['timestamp']}"
            self.history_list.addItem(item_text)

    def on_history_item_clicked(self, item):
        """历史记录项点击事件"""
        # 这里可以实现在历史记录中点击后重新加载图片的功能
        pass

    def clear_history(self):
        """清空历史记录"""
        reply = QMessageBox.question(self, '确认清空', '确定要清空所有历史记录吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_manager.history = []
            self.history_manager.save_history()
            self.refresh_history()

    def show_error(self, message):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
        print(f"错误: {message}")


def main():
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"未捕获的异常: {error_msg}")

        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("程序遇到严重错误")
            msg.setInformativeText("请查看控制台获取详细信息。")
            msg.setWindowTitle("错误")
            msg.exec_()
        except:
            pass

    sys.excepthook = exception_handler

    app = QApplication(sys.argv)
    app.setApplicationName("垃圾分类检测系统")
    app.setApplicationVersion("2.1")

    try:
        window = GarbageClassifierApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()