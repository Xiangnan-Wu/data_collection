import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from resnet import ResNet2ContinuousAction

class RobotActionDataset(Dataset):
    """机器人动作数据集"""
    def __init__(self, image_paths, action_sequences, transform=None):
        """
        初始化数据集
        
        参数:
            image_paths (list): 图像文件路径列表
            action_sequences (numpy.ndarray): 动作序列数组，形状为[N, sequence_length, 7]
            transform (callable, optional): 图像变换
        """
        self.image_paths = image_paths
        self.action_sequences = action_sequences
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取对应的动作序列
        action_sequence = torch.tensor(self.action_sequences[idx], dtype=torch.float32)
        
        return image, action_sequence


def create_dataloaders(data_dir, batch_size=32, val_split=0.2, sequence_length=5):
    """
    创建训练和验证数据加载器
    
    参数:
        data_dir (str): 数据目录路径
        batch_size (int): 批量大小
        val_split (float): 验证集比例
        sequence_length (int): 动作序列长度
        
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    # 注意：这里假设数据已经按照特定格式存储
    # 实际使用时需要根据您的数据格式进行修改
    
    # 示例：假设图像存储在 data_dir/images/ 目录下
    # 动作序列存储在 data_dir/actions.npy 文件中
    
    image_dir = os.path.join(data_dir, 'images')
    action_file = os.path.join(data_dir, 'actions.npy')
    
    # 获取所有图像文件路径
    image_paths = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(image_dir, filename))
    
    # 加载动作序列数据
    try:
        action_sequences = np.load(action_file)
        
        # 确保动作序列形状正确
        if len(action_sequences.shape) == 2:  # [N, 7]
            # 如果是单个动作，扩展为序列
            action_sequences = np.expand_dims(action_sequences, axis=1)
            action_sequences = np.repeat(action_sequences, sequence_length, axis=1)
        elif len(action_sequences.shape) == 3:  # [N, sequence_length, 7]
            # 已经是序列格式
            pass
        else:
            raise ValueError(f"不支持的动作数据形状: {action_sequences.shape}")
        
    except FileNotFoundError:
        print(f"警告：找不到动作数据文件 {action_file}")
        print("创建随机动作数据用于演示...")
        # 创建随机动作数据用于演示
        action_sequences = np.random.randn(len(image_paths), sequence_length, 7)
        # 将最后一维（夹爪状态）二值化
        action_sequences[:, :, 6] = np.random.randint(0, 2, size=(len(image_paths), sequence_length))
    
    # 确保图像和动作数量一致
    assert len(image_paths) == len(action_sequences), "图像数量和动作序列数量不匹配"
    
    # 划分训练集和验证集
    indices = np.random.permutation(len(image_paths))
    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_image_paths = [image_paths[i] for i in train_indices]
    train_action_sequences = action_sequences[train_indices]
    
    val_image_paths = [image_paths[i] for i in val_indices]
    val_action_sequences = action_sequences[val_indices]
    
    # 创建数据集
    train_dataset = RobotActionDataset(train_image_paths, train_action_sequences, transform)
    val_dataset = RobotActionDataset(val_image_paths, val_action_sequences, transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, args):
    """
    训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        args: 训练参数
    
    返回:
        训练后的模型和训练历史
    """
    device = args.device
    model.to(device)
    
    # 定义损失函数
    # 对前6个参数（位置和姿态）使用MSE损失，对第7个参数（夹爪状态）使用BCE损失
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'pose_loss': [],
        'gripper_loss': []
    }
    
    # 最佳模型保存
    best_val_loss = float('inf')
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "feature_extractor": args.feature_extractor,
                "pretrained": args.pretrained,
                "sequence_length": args.sequence_length,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "gripper_weight": args.gripper_weight,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "clip_grad": args.clip_grad,
                "device": args.device
            }
        )
        # 记录模型架构
        wandb.watch(model, log="all", log_freq=100)
    
    print("开始训练...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_pose_loss = 0.0
        train_gripper_loss = 0.0
        
        for batch_idx, (images, action_sequences) in enumerate(train_loader):
            images = images.to(device)
            action_sequences = action_sequences.to(device)
            
            # 前向传播，获取action_logits和actions
            action_logits, actions = model(images)
            
            # 计算损失
            # 对动作参数使用MSE损失
            pose_loss = mse_loss(action_logits[:, :, :6], action_sequences[:, :, :6])
            # 对夹爪状态使用BCE损失
            gripper_loss = bce_loss(action_logits[:, :, 6], action_sequences[:, :, 6])
            loss = pose_loss + args.gripper_weight * gripper_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            train_pose_loss += pose_loss.item()
            train_gripper_loss += gripper_loss.item()
            
            # 记录每个批次的损失到wandb
            if args.use_wandb and batch_idx % args.wandb_log_interval == 0:
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "batch_loss": loss.item(),
                    "batch_pose_loss": pose_loss.item(),
                    "batch_gripper_loss": gripper_loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_pose_loss /= len(train_loader)
        train_gripper_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_pose_loss = 0.0
        val_gripper_loss = 0.0
        
        # 用于记录预测结果的示例
        example_images = []
        example_true_actions = []
        example_pred_actions = []
        
        with torch.no_grad():
            for batch_idx, (images, action_sequences) in enumerate(val_loader):
                images = images.to(device)
                action_sequences = action_sequences.to(device)
                
                # 前向传播，获取action_logits和actions
                action_logits, actions = model(images)
                
                # 计算损失
                pose_loss = mse_loss(action_logits[:, :, :6], action_sequences[:, :, :6])
                gripper_loss = bce_loss(action_logits[:, :, 6], action_sequences[:, :, 6])
                loss = pose_loss + args.gripper_weight * gripper_loss
                
                # 累计损失
                val_loss += loss.item()
                val_pose_loss += pose_loss.item()
                val_gripper_loss += gripper_loss.item()
                
                # 保存一些示例用于可视化
                if batch_idx == 0 and args.use_wandb:
                    # 只取前几个样本
                    for i in range(min(3, images.size(0))):
                        example_images.append(images[i].cpu())
                        example_true_actions.append(action_sequences[i].cpu().numpy())
                        example_pred_actions.append(actions[i].cpu().numpy())
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_pose_loss /= len(val_loader)
        val_gripper_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pose_loss'].append(val_pose_loss)
        history['gripper_loss'].append(val_gripper_loss)
        
        # 记录每个epoch的损失到wandb
        if args.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pose_loss": train_pose_loss,
                "val_pose_loss": val_pose_loss,
                "train_gripper_loss": train_gripper_loss,
                "val_gripper_loss": val_gripper_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # 添加一些预测结果的可视化
            if example_images and example_pred_actions and example_true_actions:
                # 将图像转换回正常范围用于显示
                for i in range(len(example_images)):
                    img = example_images[i].permute(1, 2, 0).numpy()
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    
                    # 创建预测vs真实动作的表格
                    table_data = []
                    for t in range(len(example_pred_actions[i])):
                        pred = example_pred_actions[i][t]
                        true = example_true_actions[i][t]
                        table_data.append([
                            t+1,  # 时间步
                            f"{pred[0]:.4f}", f"{true[0]:.4f}",  # delta x
                            f"{pred[1]:.4f}", f"{true[1]:.4f}",  # delta y
                            f"{pred[2]:.4f}", f"{true[2]:.4f}",  # delta z
                            f"{pred[3]:.4f}", f"{true[3]:.4f}",  # delta r
                            f"{pred[4]:.4f}", f"{true[4]:.4f}",  # delta p
                            f"{pred[5]:.4f}", f"{true[5]:.4f}",  # delta y
                            f"{'打开' if pred[6] > 0.5 else '关闭'}", f"{'打开' if true[6] > 0.5 else '关闭'}"  # gripper
                        ])
                    
                    # 记录到wandb
                    log_dict[f"example_{i+1}_predictions"] = wandb.Table(
                        columns=["时间步", "预测dx", "真实dx", "预测dy", "真实dy", "预测dz", "真实dz", 
                                "预测dr", "真实dr", "预测dp", "真实dp", "预测dy", "真实dy", 
                                "预测夹爪", "真实夹爪"],
                        data=table_data
                    )
            
            wandb.log(log_dict)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
            
            # 在wandb中记录最佳模型
            if args.use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
                # 保存最佳模型到wandb
                wandb.save(os.path.join(args.output_dir, 'best_model.pth'))
        
        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | 时间: {epoch_time:.2f}s | 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"训练损失: {train_loss:.4f} (姿态: {train_pose_loss:.4f}, 夹爪: {train_gripper_loss:.4f})")
        print(f"验证损失: {val_loss:.4f} (姿态: {val_pose_loss:.4f}, 夹爪: {val_gripper_loss:.4f})")
        print("-" * 80)
        
        # 每隔几个epoch保存一次检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            # 在wandb中保存检查点
            if args.use_wandb:
                wandb.save(checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.npy')
    np.save(history_path, history)
    
    # 在wandb中保存最终模型和训练历史
    if args.use_wandb:
        wandb.save(final_model_path)
        wandb.save(history_path)
        
        # 记录最终指标
        wandb.run.summary["final_train_loss"] = train_loss
        wandb.run.summary["final_val_loss"] = val_loss
        wandb.run.summary["total_epochs"] = args.epochs
        
        # 结束wandb运行
        wandb.finish()
    
    return model, history


def plot_training_history(history, output_dir):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('总损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制姿态和夹爪损失
    plt.subplot(1, 2, 2)
    plt.plot(history['pose_loss'], label='姿态损失')
    plt.plot(history['gripper_loss'], label='夹爪损失')
    plt.title('组件损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练ResNet2ContinuousAction模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    
    # 模型参数
    parser.add_argument('--feature_extractor', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'], help='特征提取器类型')
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--sequence_length', type=int, default=5, help='动作序列长度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--gripper_weight', type=float, default=1.0, help='夹爪损失权重')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--save_interval', type=int, default=5, help='保存检查点的间隔')
    
    # wandb参数
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录训练过程')
    parser.add_argument('--wandb_project', type=str, default='resnet2action', help='wandb项目名称')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb运行名称')
    parser.add_argument('--wandb_log_interval', type=int, default=10, help='wandb记录间隔(批次)')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果未指定wandb运行名称，则自动生成一个
    if args.use_wandb and args.wandb_run_name is None:
        args.wandb_run_name = f"{args.feature_extractor}_seq{args.sequence_length}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        val_split=args.val_split,
        sequence_length=args.sequence_length
    )
    
    # 创建模型
    model = ResNet2ContinuousAction(
        feature_extractor_type=args.feature_extractor,
        pretrained=args.pretrained,
        sequence_length=args.sequence_length,
        device=args.device
    )
    
    # 训练模型
    model, history = train_model(model, train_loader, val_loader, args)
    
    # 绘制训练历史
    plot_training_history(history, args.output_dir)
    
    print(f"训练完成！模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()