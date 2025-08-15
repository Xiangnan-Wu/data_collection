import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class BasicBlock(nn.Module):
    """ResNet的基本残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet的瓶颈残差块，用于更深层次的ResNet模型"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet模型"""
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播，返回特征向量"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        return features


class ResNetFeatureExtractor:
    """使用ResNet作为特征提取器的类"""
    def __init__(self, model_type='resnet18', pretrained=True, device='cuda'):
        """
        初始化ResNet特征提取器
        
        参数:
            model_type (str): 使用的ResNet模型类型，可选'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            pretrained (bool): 是否使用预训练权重
            device (str): 运行设备，'cuda'或'cpu'
        """
        self.device = device
        
        # 使用torchvision预训练模型
        if pretrained:
            if model_type == 'resnet18':
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif model_type == 'resnet34':
                self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif model_type == 'resnet50':
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            elif model_type == 'resnet101':
                self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            elif model_type == 'resnet152':
                self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
        else:
            # 使用自定义实现的ResNet
            if model_type == 'resnet18':
                base_model = ResNet(BasicBlock, [2, 2, 2, 2])
            elif model_type == 'resnet34':
                base_model = ResNet(BasicBlock, [3, 4, 6, 3])
            elif model_type == 'resnet50':
                base_model = ResNet(Bottleneck, [3, 4, 6, 3])
            elif model_type == 'resnet101':
                base_model = ResNet(Bottleneck, [3, 4, 23, 3])
            elif model_type == 'resnet152':
                base_model = ResNet(Bottleneck, [3, 8, 36, 3])
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            self.model = base_model
        
        # 移除分类层
        if pretrained:
            self.feature_dim = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            if model_type.startswith('resnet'):
                if '18' in model_type or '34' in model_type:
                    self.feature_dim = 512
                else:
                    self.feature_dim = 2048
        
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
    
    def extract_features(self, images):
        """
        从图像中提取特征
        
        参数:
            images (torch.Tensor): 形状为[B, C, H, W]的图像张量
            
        返回:
            features (torch.Tensor): 形状为[B, feature_dim]的特征向量
        """
        with torch.no_grad():
            images = images.to(self.device)
            features = self.model(images)
            if len(features.shape) == 4:  # 如果特征是[B, C, H, W]格式
                features = torch.flatten(features, 1)
            return features


class ResNet2ContinuousAction(nn.Module):
    """基于ResNet特征提取的连续动作预测模型"""
    def __init__(self, feature_extractor_type='resnet18', pretrained=True, sequence_length=5, device='cuda'):
        """
        初始化连续动作预测模型
        
        参数:
            feature_extractor_type (str): 特征提取器类型
            pretrained (bool): 是否使用预训练权重
            sequence_length (int): 预测的连续动作序列长度
            device (str): 运行设备
        """
        super(ResNet2ContinuousAction, self).__init__()
        self.device = device
        self.sequence_length = sequence_length
        
        # 特征提取器
        self.feature_extractor = ResNetFeatureExtractor(
            model_type=feature_extractor_type,
            pretrained=pretrained,
            device=device
        )
        
        # 获取特征维度
        feature_dim = self.feature_extractor.feature_dim
        
        # 特征转换层，将单个特征向量转换为序列特征
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # LSTM层用于序列预测
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7个输出：delta x, delta y, delta z, delta r, delta p, delta y, gripper state
        )
        
        # 将模型移至指定设备
        self.to(device)
    
    def forward(self, x):
        """
        前向传播函数，输入单张图像，输出连续的动作序列
        
        参数:
            x: 输入图像张量，形状为 [batch_size, 3, height, width]
            
        返回:
            actions: 预测的动作序列，形状为 [batch_size, sequence_length, 7]
                     每个动作包含7个参数：delta x, delta y, delta z, delta r, delta p, delta y, gripper state
        """
        # 提取图像特征
        features = self.feature_extractor.extract_features(x)  # [batch_size, feature_dim]
        
        # 特征转换
        transformed_features = self.feature_transform(features)  # [batch_size, 512]
        
        # 创建初始隐藏状态和单元状态
        batch_size = features.size(0)
        h0 = torch.zeros(2, batch_size, 256).to(self.device)  # 2层LSTM
        c0 = torch.zeros(2, batch_size, 256).to(self.device)
        
        # 扩展特征为序列形式，每个时间步都使用相同的特征
        # [batch_size, 1, 512] -> [batch_size, sequence_length, 512]
        seq_features = transformed_features.unsqueeze(1).expand(-1, self.sequence_length, -1)
        
        # 通过LSTM处理序列
        lstm_out, _ = self.lstm(seq_features, (h0, c0))  # [batch_size, sequence_length, 256]
        
        # 预测每个时间步的动作
        actions_seq = []
        for t in range(self.sequence_length):
            action_t = self.action_head(lstm_out[:, t, :])  # [batch_size, 7]
            actions_seq.append(action_t)
        
        # 堆叠所有时间步的动作
        action_logits = torch.stack(actions_seq, dim=1)  # [batch_size, sequence_length, 7
        
        return action_logits
    
    def predict_sequence(self, image):
        """
        预测单个图像的连续动作序列
        
        参数:
            image: 输入图像张量，形状为 [1, 3, height, width]
            
        返回:
            action_sequence: 预测的动作序列，形状为 [sequence_length, 7]
                            每个动作包含7个参数：delta x, delta y, delta z, delta r, delta p, delta y, gripper state
        """
        self.eval()
        with torch.no_grad():
            # 前向传播
            action_logits= self(image)  # [1, sequence_length, 7]
            
            # 将gripper state二值化（大于0.5为1，否则为0）
            actions_np = action_logits[0].cpu().numpy()  # [sequence_length, 7]
            actions_np[:, 6] = np.where(actions_np[:, 6] > 0, 1, 0)
            
            return actions_np


# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建LSTM模型
    lstm_model = ResNet2ContinuousAction(
        feature_extractor_type='resnet18',
        pretrained=False,
        sequence_length=5,
        device=device
    )
    
    
    # 创建一个示例输入 (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    
    # LSTM模型前向传播
    lstm_actions = lstm_model(x)
    print(f"输入形状: {x.shape}")
    print(f"LSTM模型输出动作序列形状: {lstm_actions.shape}")  # 应该是 [2, 5, 7]
    
    
    # 单个图像预测示例
    single_image = torch.randn(1, 3, 224, 224)
    
    # LSTM模型预测
    lstm_action_sequence = lstm_model.predict_sequence(single_image)
    print(f"\nLSTM模型预测的动作序列形状: {lstm_action_sequence.shape}")  # 应该是 [5, 7]
    
    # 打印第一个动作
    print(f"\n第一个动作 (LSTM):")
    print(f"位置增量: [{lstm_action_sequence[0][0]:.4f}, {lstm_action_sequence[0][1]:.4f}, {lstm_action_sequence[0][2]:.4f}]")
    print(f"姿态增量: [{lstm_action_sequence[0][3]:.4f}, {lstm_action_sequence[0][4]:.4f}, {lstm_action_sequence[0][5]:.4f}]")
    print(f"夹爪状态: {'打开' if lstm_action_sequence[0][6] == 1 else '关闭'}")
    
    # 打印整个动作序列
    print(f"\n完整动作序列 (LSTM):")
    for i, action in enumerate(lstm_action_sequence):
        print(f"步骤 {i+1}: 位置[{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}], "
              f"姿态[{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}], "
              f"夹爪{'打开' if action[6] == 1 else '关闭'}")
