import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score

# -----------------------
# 数据加载与缓存
# -----------------------
@st.cache_data
def load_data():
    train = pd.read_csv("sign_mnist_train.csv")
    test = pd.read_csv("sign_mnist_test.csv")
    return train, test

train, test = load_data()

# -----------------------
# 数据预处理
# -----------------------
X_train = torch.tensor(train.drop("label", axis=1).values / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(train["label"].values, dtype=torch.long)

X_test = torch.tensor(test.drop("label", axis=1).values / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test = torch.tensor(test["label"].values, dtype=torch.long)

letters = [chr(i) for i in range(65, 91)]

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧏 手语字母识别 (Sign Language MNIST) — PyTorch 版")
st.write("通过视觉化与深度学习探索手语数据集")

# 交互控件
selected_letters = st.multiselect("选择要显示的手势字母（可多选）", letters, default=["A", "B"])
num_images = st.slider("每个字母显示图片数", 1, 10, 5)

# -----------------------
# 显示示例图片
# -----------------------
st.subheader("🔹 手势示例图片")
for letter in selected_letters:
    label_idx = ord(letter) - 65
    imgs = X_train[y_train == label_idx][:num_images]
    st.write(f"字母 {letter} 示例：")
    cols = st.columns(num_images)
    for i, img in enumerate(imgs):
        cols[i].image(img.squeeze().numpy(), width=100)

# -----------------------
# EDA 可视化
# -----------------------
st.subheader("📊 每个字母手势数量统计")
fig, ax = plt.subplots(figsize=(10,4))
sns.countplot(x="label", data=train, ax=ax, palette="coolwarm")
ax.set_xlabel("字母标签 (0=A, 25=Z)")
ax.set_ylabel("数量")
st.pyplot(fig)

st.subheader("📈 像素强度分布示例")
plt.hist(X_train[0].flatten().numpy(), bins=20, color="purple")
plt.xlabel("像素强度")
plt.ylabel("频数")
st.pyplot(plt)

# 平均手势热力图
st.subheader("🔥 平均手势图像（字母 A 示例）")
avg_img = X_train[y_train == 0].mean(dim=0).squeeze()
sns.heatmap(avg_img.numpy(), cmap="viridis")
st.pyplot(plt)

# -----------------------
# PyTorch 模型定义
# -----------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# 模型训练
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs=3, batch_size=128, lr=0.001):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    train_acc_list, val_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        # 测试集准确率
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(y.cpu().numpy())
        val_acc = accuracy_score(y_true, y_pred)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        st.write(f"Epoch {epoch+1}/{epochs} | 训练准确率: {train_acc:.2f} | 测试准确率: {val_acc:.2f}")

    return model, train_acc_list, val_acc_list

# -----------------------
# 训练按钮
# -----------------------
st.subheader("🧩 PyTorch CNN 训练")
if st.button("开始训练模型 (3 epochs)"):
    model, train_acc_list, val_acc_list = train_model()

    # 绘制训练曲线
    fig, ax = plt.subplots()
    ax.plot(train_acc_list, label="训练准确率")
    ax.plot(val_acc_list, label="测试准确率")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("准确率")
    ax.legend()
    st.pyplot(fig)
    st.success(f"最终测试准确率：{val_acc_list[-1]:.2f}")
