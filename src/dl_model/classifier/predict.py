import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from model import shufflenet_v2_x2_0

test_image_path = '../../data/classify/RESIDE-IN/1400_1.png'
dictclass = ['GT-RAIN', 'RESIDE-IN', 'RESIDE-OUT']

model = shufflenet_v2_x2_0()
model.load_state_dict(torch.load('./best.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加批次维度
    return image.to(device)

# 进行预测
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

predicted_class = predict(test_image_path)
print('input_path: ', test_image_path)
print(f'Predicted class: {dictclass[predicted_class]}')
