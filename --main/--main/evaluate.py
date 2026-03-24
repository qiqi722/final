import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image

print("=== 步骤3: 模型评估 ===")

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略字体警告和torchvision弃用警告
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 加载测试数据
print("1. 加载测试数据...")
X_test = np.load('data/processed/splits/X_test.npy')
y_test = np.load('data/processed/splits/y_test.npy')

# 定义与训练时相同的预处理
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)
        
        # 关键修复：将标签转换为long类型
        label = torch.tensor(label, dtype=torch.long)
            
        return image, label

# 2. 加载模型
print("2. 加载训练好的模型...")

class EfficientGarbageClassifier(nn.Module):
    def __init__(self, num_classes=5):  # 修正：改为5类
        super(EfficientGarbageClassifier, self).__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 冻结前几层 - 与训练时一致
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False

        # 修改最后的全连接层 - 与训练时完全一致
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

def main():
    # 创建测试数据集
    test_dataset = CustomDataset(X_test, y_test, transform=test_transform)
    # 禁用多进程数据加载
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型实例
    model = EfficientGarbageClassifier(num_classes=5)  # 修正：改为5类
    model = model.to(device)

    # 加载训练好的权重
    print("加载模型权重...")
    checkpoint = torch.load('models/best_model.pth', map_location=device, weights_only=False)

    # 更灵活的权重加载
    model_state_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']

    # 只加载匹配的键
    matched_keys = []
    unmatched_keys = []
    for name, param in pretrained_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name] = param
                matched_keys.append(name)
            else:
                unmatched_keys.append(f"形状不匹配: {name} - 模型: {model_state_dict[name].shape}, 检查点: {param.shape}")
        else:
            unmatched_keys.append(f"缺失键: {name}")

    print(f"成功匹配 {len(matched_keys)}/{len(model_state_dict)} 个参数")
    if unmatched_keys:
        print("不匹配的参数:")
        for key in unmatched_keys[:10]:  # 只显示前10个
            print(f"  {key}")

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    print("✅ 模型准备完成")

    # 3. 在测试集上评估
    print("3. 在测试集上评估...")
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    test_loss /= len(test_loader)

    # 4. 计算指标
    print("4. 计算评估指标...")
    accuracy = accuracy_score(all_targets, all_predictions)

    # 加载类别信息
    with open('data/processed/dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    classes = dataset_info['classes']

    # 计算详细指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, labels=range(len(classes))
    )

    # 计算宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted'
    )

    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {accuracy:.4f}")
    print(f"宏平均 F1: {f1_macro:.4f}")
    print(f"加权平均 F1: {f1_weighted:.4f}")

    print("\n详细分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=classes))

    # 5. 绘制混淆矩阵
    print("5. 绘制混淆矩阵...")
    cm = confusion_matrix(all_targets, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': '样本数量'})
    plt.title('Confusion Matrix - 垃圾分类模型', fontsize=16, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵已保存到 models/confusion_matrix.png")
    plt.show()

    # 6. 绘制类别性能对比图
    print("6. 绘制类别性能对比图...")
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=classes)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25

    plt.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    plt.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

    plt.xlabel('垃圾类别')
    plt.ylabel('分数')
    plt.title('各类别性能对比')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/class_performance.png', dpi=300, bbox_inches='tight')
    print("类别性能对比图已保存到 models/class_performance.png")
    plt.show()

    # 7. 计算每个类别的准确率
    print("7. 计算各类别准确率...")
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    class_confidences = [[] for _ in range(len(classes))]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                class_confidences[label].append(confidences[i].item())

    class_accuracies = {}
    class_avg_confidences = {}
    for i in range(len(classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            avg_confidence = np.mean(class_confidences[i]) * 100 if class_confidences[i] else 0
            class_accuracies[classes[i]] = round(accuracy, 2)
            class_avg_confidences[classes[i]] = round(avg_confidence, 2)
            print(f'{classes[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]}), 平均置信度: {avg_confidence:.2f}%')

    # 8. 生成KPI报告
    print("8. 生成KPI报告...")
    
    # 修复：安全地获取测试集类别分布
    try:
        # 尝试从不同可能的位置获取测试集分布
        if 'class_distribution' in dataset_info and 'test' in dataset_info['class_distribution']:
            class_distribution = dataset_info['class_distribution']['test']
        elif 'class_distribution' in dataset_info and 'augmented' in dataset_info['class_distribution']:
            class_distribution = dataset_info['class_distribution']['augmented']['test']
        elif 'sizes' in dataset_info and 'test' in dataset_info['sizes']:
            # 如果没有类别分布，至少记录测试集大小
            class_distribution = {"total": dataset_info['sizes']['test']}
        else:
            # 如果都没有，则使用我们计算的实际分布
            class_distribution = {classes[i]: class_total[i] for i in range(len(classes))}
    except KeyError:
        # 如果出现任何错误，使用我们计算的实际分布
        class_distribution = {classes[i]: class_total[i] for i in range(len(classes))}
    
    kpi_report = {
        'test_accuracy': float(accuracy),
        'test_loss': float(test_loss),
        'test_size': len(y_test),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'class_distribution': class_distribution,
        'class_accuracies': class_accuracies,
        'class_confidences': class_avg_confidences,
        'class_metrics': {
            'precision': {classes[i]: float(precision[i]) for i in range(len(classes))},
            'recall': {classes[i]: float(recall[i]) for i in range(len(classes))},
            'f1_score': {classes[i]: float(f1[i]) for i in range(len(classes))}
        },
        'best_validation_accuracy': float(checkpoint.get('val_acc', 0)),
        'model_architecture': 'ResNet18',
        'training_epoch': checkpoint.get('epoch', 0),
        'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print("\n=== KPI 报告 ===")
    print(f"测试准确率: {kpi_report['test_accuracy']:.4f}")
    print(f"测试损失: {kpi_report['test_loss']:.4f}")
    print(f"宏平均 F1: {kpi_report['macro_f1']:.4f}")
    print(f"最佳验证准确率: {kpi_report['best_validation_accuracy']:.4f}")
    print("\n各类别准确率:")
    for class_name, acc in kpi_report['class_accuracies'].items():
        conf = kpi_report['class_confidences'].get(class_name, 0)
        print(f"  {class_name}: {acc}% (平均置信度: {conf}%)")

    with open('models/kpi_report.json', 'w', encoding='utf-8') as f:
        json.dump(kpi_report, f, indent=2, ensure_ascii=False)

    print("\nKPI报告已保存到 models/kpi_report.json")
    print("评估完成! 🎉")

    # 9. 生成性能总结
    print("\n=== 性能总结 ===")
    print(f"总体准确率: {accuracy:.2%}")
    print(f"模型鲁棒性: {'优秀' if f1_macro > 0.85 else '良好' if f1_macro > 0.75 else '一般'}")
    print(f"类别平衡性: {'良好' if min(precision) > 0.7 and min(recall) > 0.7 else '需要改进'}")

    # 找出表现最好和最差的类别
    best_class = classes[np.argmax(f1)]
    worst_class = classes[np.argmin(f1)]
    print(f"表现最佳类别: {best_class} (F1: {np.max(f1):.3f})")
    print(f"表现最差类别: {worst_class} (F1: {np.min(f1):.3f})")

if __name__ == '__main__':
    main()