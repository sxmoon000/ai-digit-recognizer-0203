"""
手写数字识别 — scikit-learn MLP 神经网络
使用 MNIST 数据集训练一个多层感知机识别手写数字 0-9。
技术栈：多层感知机 (MLP) + 特征标准化 + 混淆矩阵
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path

print("[1/4] Loading MNIST...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
X = X / 255.0  # 归一化
y = y.astype(int)
print(f"   样本数: {len(X)}, 特征数: 784 (28x28), 类别: 0-9")

print("[2/4] Splitting & Scaling...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("[3/4] Training MLP (hidden=128,128)...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=30, random_state=42, verbose=False)
mlp.fit(X_train, y_train)
acc = accuracy_score(y_test, mlp.predict(X_test))
print(f"   测试准确率: {acc:.2%}")

print("[4/4] Saving model...")
Path("model").mkdir(exist_ok=True)
joblib.dump(mlp, "model/mlp_mnist.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Demo
print("\n🧪 Demo: 前10个测试样本预测")
preds = mlp.predict(X_test[:10])
for i in range(10):
    print(f"   真实: {y_test[i]}, 预测: {preds[i]}, {'✓' if y_test[i]==preds[i] else '✗'}")

print(f"\n✅ 完成! 准确率: {acc:.2%}")
