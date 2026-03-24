import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from torchvision import models, transforms
import json
from sklearn.utils.class_weight import compute_class_weight

# 将类定义和函数定义放在 if __name__ == '__main__' 外面
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
            # 默认转换
            image = torch.FloatTensor(image).permute(2, 0, 1)
            
        # 关键修复：将标签转换为long类型
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

class EfficientGarbageClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientGarbageClassifier, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结前几层
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False

        # 修改最后的全连接层
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

def validate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return val_loss / len(val_loader), correct / total

if __name__ == '__main__':
    print("=== 垃圾分类模型训练 ===")
    
    # 确保models目录存在
    os.makedirs('models', exist_ok=True)

    # 1. 加载数据
    print("1. 加载预处理的数据...")
    X_train = np.load('data/processed/splits/X_train.npy')
    y_train = np.load('data/processed/splits/y_train.npy')
    X_val = np.load('data/processed/splits/X_val.npy')
    y_val = np.load('data/processed/splits/y_val.npy')

    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")

    # 2. 数据增强和预处理
    print("2. 准备数据增强和数据加载器...")

    # 定义数据增强变换
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(X_val, y_val, transform=val_transform)

    # 计算类别权重
    print("计算类别权重...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights}")

    # 创建数据加载器 - 在Windows上建议使用num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 3. 创建模型
    print("3. 创建模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = EfficientGarbageClassifier(num_classes=5)
    model = model.to(device)

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    print(f"冻结参数比例: {(total_params - trainable_params) / total_params:.2%}")

    # 4. 训练模型
    print("4. 开始训练...")
    # 使用带权重的损失函数处理类别不平衡
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用AdamW

    # 学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    best_val_acc = 0.0
    patience = 15  # 增加耐心值
    patience_counter = 0

    start_time = time.time()

    print("开始训练循环...")
    for epoch in range(100):  # 增加最大epoch数
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # 验证阶段
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'class_weights': class_weights,
                'train_acc': train_acc
            }, 'models/best_model.pth')
            print(f"✅ Epoch {epoch + 1}: 保存最佳模型，验证准确率: {val_acc:.4f}")
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= patience:
            print(f"🛑 早停触发! 在 epoch {epoch + 1}")
            break

        # 打印进度
        if (epoch + 1) % 2 == 0:  # 更频繁的打印
            print(f'Epoch {epoch + 1}/100:')
            print(f'  训练: 损失={train_loss:.4f}, 准确率={train_acc:.4f}')
            print(f'  验证: 损失={val_loss:.4f}, 准确率={val_acc:.4f}')
            print(f'  学习率: {current_lr:.6f}, 早停: {patience_counter}/{patience}')

    training_time = time.time() - start_time
    print(f"训练完成! 用时: {training_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 5. 绘制训练曲线
    print("5. 绘制训练曲线...")
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='red', alpha=0.7, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue', alpha=0.7, linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.7, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True, alpha=0.3)

    # 标记最佳准确率
    best_epoch = np.argmax(val_accuracies)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    plt.text(best_epoch, 0.5, f'best\n{val_accuracies[best_epoch]:.3f}',
             rotation=0, transform=plt.gca().get_xaxis_transform(),
             ha='center', va='center')

    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存到 models/training_history.png")
    plt.show()

    # 6. 加载最佳模型进行最终测试
    print("6. 加载最佳模型进行最终评估...")
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 在验证集上最终评估
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算最终准确率
    final_accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"最终验证准确率: {final_accuracy:.4f}")

    # 7. 保存训练总结
    training_summary = {
        'best_validation_accuracy': float(best_val_acc),
        'final_validation_accuracy': float(final_accuracy),
        'total_training_time': float(training_time),
        'total_epochs_trained': len(train_losses),
        'best_epoch': int(best_epoch),
        'model_architecture': 'ResNet18',
        'num_classes': 5,  # 明确记录类别数
        'image_size': [224, 224],
        'device_used': str(device),
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'class_weights': class_weights.tolist(),
        'training_parameters': {
            'batch_size': 16,
            'initial_learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR (T_max=50)',
            'loss_function': 'CrossEntropyLoss with class weights',
            'early_stopping_patience': patience,
            'data_augmentation': True,
            'gradient_clipping': True
        }
    }

    with open('models/training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)

    print("训练总结已保存到 models/training_summary.json")
    print("🎉 模型训练完成!")