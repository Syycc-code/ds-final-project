import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# -------------------
# 数据加载
# -------------------
@st.cache_data
def load_data():
    train = pd.read_csv("sign_mnist_train.csv")
    test = pd.read_csv("sign_mnist_test.csv")
    return train, test

train, test = load_data()

# -------------------
# 数据预处理
# -------------------
X_train = train.drop("label", axis=1).values.reshape(-1,28,28,1)/255.0
y_train = to_categorical(train["label"], num_classes=26)

X_test = test.drop("label", axis=1).values.reshape(-1,28,28,1)/255.0
y_test = to_categorical(test["label"], num_classes=26)

letters = [chr(i) for i in range(65, 91)]  # A-Z 字母

# -------------------
# App 标题
# -------------------
st.title("手语字母 Sign Language MNIST 数据分析")
st.write("一个交互式数据分析与简单分类演示")

# -------------------
# 交互控件 1：选择显示的字母
# -------------------
selected_letters = st.multiselect("选择要显示的手势字母（可多选）", letters, default=['A', 'B'])

# 交互控件 2：选择显示图片数量
num_images = st.slider("每个字母显示的图片数量", min_value=1, max_value=10, value=5)

# -------------------
# 显示示例图片
# -------------------
st.subheader("手势示例图片")
for letter in selected_letters:
    label_idx = ord(letter) - 65
    sample_images = X_train[train["label"]==label_idx][:num_images]
    st.write(f"字母 {letter} 示例:")
    cols = st.columns(num_images)
    for i, img in enumerate(sample_images):
        cols[i].image(img.reshape(28,28), width=100)

# -------------------
# EDA 图表
# -------------------
st.subheader("EDA：每个字母手势数量统计")
plt.figure(figsize=(12,5))
sns.countplot(x="label", data=train)
plt.xlabel("字母标签 (0=A, 25=Z)")
plt.ylabel("数量")
st.pyplot(plt)

st.subheader("EDA：像素强度分布示例")
plt.hist(X_train[0].flatten(), bins=20, color='purple')
plt.xlabel("像素强度")
plt.ylabel("频数")
st.pyplot(plt)

# 平均手势图像（热力图）
st.subheader("EDA：每个字母的平均手势图像")
letter_avg = X_train[train["label"]==0].mean(axis=0).reshape(28,28)  # 默认 A
st.write("字母 A 平均手势图像示例")
sns.heatmap(letter_avg, cmap="viridis")
st.pyplot(plt)

# -------------------
# CNN 模型训练（可选）
# -------------------
st.subheader("基础 CNN 分类模型训练演示")
if st.button("训练 CNN 模型 (3 epochs)"):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    st.write("开始训练...")
    history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), verbose=0)
    st.success("训练完成！")
    val_acc = history.history['val_accuracy'][-1]
    st.write(f"模型在测试集上的准确率: {val_acc:.2f}")
    
    # 可视化训练过程
    st.subheader("训练过程图")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='训练准确率')
    ax.plot(history.history['val_accuracy'], label='验证准确率')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.legend()
    st.pyplot(fig)
