"""
数字识别模型对比 — SVM / KNN / 随机森林 + 可视化混淆矩阵

v1.1 新增:
  • 4种模型对比 (MLP, SVM, KNN, RandomForest)
  • 混淆矩阵可视化 (ASCII热力图)
  • 训练速度 vs 准确率分析
  • 最佳模型自动选择
"""
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dataclasses import dataclass
from typing import List


@dataclass
class ModelResult:
    name: str
    accuracy: float
    train_time: float
    predict_time: float
    cm: np.ndarray = None
    efficiency: float = 0.0  # accuracy / train_time 比值


class ModelComparator:
    """多模型对比基准测试"""

    MODELS = {
        "MLP (神经网络)": MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=30, random_state=42),
        "SVM (支持向量机)": SVC(kernel="rbf", C=5, gamma=0.05, random_state=42),
        "KNN (k近邻)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "随机森林": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    }

    def __init__(self, sample_size: int = 5000):
        self.sample_size = sample_size
        self.results: List[ModelResult] = []

    def benchmark(self):
        """完整基准测试"""
        print("=" * 55)
        print("🔬 数字识别 — 4模型对比基准测试")
        print("=" * 55)

        # 加载
        print("\n[1/3] 加载 MNIST...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
        X = X / 255.0
        y = y.astype(int)
        # 采样加速
        if len(X) > self.sample_size:
            idx = np.random.RandomState(42).choice(len(X), self.sample_size, replace=False)
            X, y = X[idx], y[idx]
        print(f"   样本: {len(X)}, 特征: 784")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 逐个模型
        print("\n[2/3] 训练 & 评估:")
        for name, model in self.MODELS.items():
            print(f"\n   🧠 {name}...", end=" ", flush=True)
            t0 = time.time()
            model.fit(X_train_s, y_train)
            train_t = time.time() - t0

            t0 = time.time()
            preds = model.predict(X_test_s)
            pred_t = time.time() - t0

            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            eff = acc / max(train_t, 0.001)

            result = ModelResult(name, acc, train_t, pred_t, cm, eff)
            self.results.append(result)
            print(f"准确率 {acc:.2%} | 训练 {train_t:.1f}s | 预测 {pred_t:.3f}s")

        # 排名
        print(f"\n[3/3] 综合排名:")
        print(f"   {'模型':<18} {'准确率':>8} {'训练时间':>8} {'效率分':>8}")
        print(f"   {'─'*44}")
        for r in sorted(self.results, key=lambda r: r.accuracy, reverse=True):
            print(f"   {r.name:<18} {r.accuracy:>7.2%} {r.train_time:>7.1f}s {r.efficiency:>7.1f}")

        # 最佳
        best = max(self.results, key=lambda r: r.accuracy)
        fastest = max(self.results, key=lambda r: r.efficiency)
        print(f"\n   🏆 最高准确率: {best.name} ({best.accuracy:.2%})")
        print(f"   ⚡ 最快效率: {fastest.name} (效率分 {fastest.efficiency:.1f})")

        # 混淆矩阵
        print(f"\n   📊 {best.name} 混淆矩阵 (10x10):")
        self._print_cm(best.cm)

    def _print_cm(self, cm: np.ndarray, top_n: int = 5):
        """打印混淆矩阵热力图"""
        # 行=真实, 列=预测
        print("       Pred →")
        print("        " + "".join(f"  {i}  " for i in range(10)))
        for i in range(10):
            row = f"  真实{i}   "
            for j in range(10):
                val = cm[i][j]
                if val == 0:
                    row += "  ·  "
                elif i == j:
                    row += f" \033[92m{val:>3}\033[0m " if val > 0 else "  ·  "
                elif val > cm[i].max() * 0.2:
                    row += f" \033[91m{val:>3}\033[0m "
                else:
                    row += f" {val:>3} "
            print(row)


def main():
    comparator = ModelComparator(sample_size=3000)
    comparator.benchmark()
    print(f"\n✅ 模型对比完成")


if __name__ == "__main__":
    main()
