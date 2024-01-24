import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import torchvision.models as models
from torch import nn

from vit_model import vit_base_patch16_224_in21k as create_model

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
data_path = "dataset/flower_photos"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# 加载模型
model_weight_path = "./weights/resnet/model-190.pth"
# model = create_model(num_classes=5, has_logits=False).to(device)
model = torch.load(model_weight_path)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5, bias=False)

model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
model.eval()

# 获取预测结果和标签
labels = []
preds = []
for inputs, targets in dataset:
    inputs = inputs.unsqueeze(0).to(device)
    targets = torch.tensor(targets).to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    labels.append(targets.item())
    preds.append(predicted.item())

result = classification_report(labels, preds, target_names=dataset.class_to_idx, output_dict=True)
result_csv = pd.DataFrame(result).transpose()
result_csv.to_csv("./result/resnet/result.csv", index=True)
print("resulte saved as result.csv")

# 生成混淆矩阵
cm = confusion_matrix(labels, preds)
classes = dataset.classes
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# 保存为CSV文件
cm_df.to_csv("./result/resnet/confusion_matrix.csv")
print("Confusion matrix saved as confusion_matrix.csv")
