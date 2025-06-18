import os
import glob
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from model.DeepLabv3model import DeepLabV3PlusForNucleiSegmentation
from utils.dataset import MoNuSegDataset
from torch.utils.data import DataLoader

# 计算 Dice 系数
def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

# 计算 IoU
def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def load_model(model_path, device):
    """安全加载模型权重并返回已加载的模型"""
    # 初始化模型
    net = DeepLabV3PlusForNucleiSegmentation(num_classes=1)
    net.to(device=device)

    # 安全加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)

    # 设置为评估模式
    net.eval()
    return net

def visualize_prediction(image, true_mask, pred_mask):
    """可视化原始图像、真实掩码和预测掩码"""
    plt.figure(figsize=(15, 5))

    # 显示原始图像
    plt.subplot(131)
    plt.title('Original Image')
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.axis('off')

    # 显示真实掩码
    plt.subplot(132)
    plt.title('True Mask')
    true_mask = true_mask.squeeze(0).cpu().numpy()
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')

    # 显示预测掩码
    plt.subplot(133)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = 'deeplabv3plus_model.pth'  # 使用DeepLabV3+训练的模型
    DATA_PATH = r"C:\ZHyl\Dzy\data"
    THRESHOLD = 0.5  # 二值化阈值

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    net = load_model(MODEL_PATH, device)

    # 加载测试集
    test_dataset = MoNuSegDataset(DATA_PATH, mode='test', augment=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)

            # 执行预测
            outputs = net(images)
            outputs = torch.sigmoid(outputs)

            # 处理预测结果
            pred_masks = outputs.cpu().numpy()[0, 0]
            pred_masks[pred_masks >= THRESHOLD] = 1
            pred_masks[pred_masks < THRESHOLD] = 0

            # 处理真实掩码
            true_masks = masks.cpu().numpy()[0, 0]

            # 计算 Dice 系数和 IoU
            dice = dice_coefficient(true_masks, pred_masks)
            iou_score = iou(true_masks, pred_masks)

            dice_scores.append(dice)
            iou_scores.append(iou_score)

            # 随机选择一张图片进行可视化
            if i == 0:
                visualize_prediction(images[0], masks[0], pred_masks)

    # 计算平均 Dice 系数和 IoU
    average_dice = np.mean(dice_scores)
    average_iou = np.mean(iou_scores)

    print(f"平均 Dice 系数: {average_dice:.4f}")
    print(f"平均 IoU: {average_iou:.4f}")