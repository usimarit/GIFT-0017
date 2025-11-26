"""
svm_experiments.py

功能：
1. 随机二分类数据集 + SVM + 嵌套交叉验证（10-fold 外层，5-fold 内层）
2. Adult 数据集上的 SVM 训练与评估：
   - 使用全部特征
   - 使用部分特征（子集）以比较性能差异
"""

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =============== 工具函数：打印结果 ===============

def print_fold_results(results):
    """
    results: 字典，key 为 metric 名称，value 为长度为 n_folds 的列表
    """
    n_folds = len(next(iter(results.values())))
    print("\n=== 每折结果 ===")
    header = "Fold\t" + "\t".join(results.keys())
    print(header)
    for i in range(n_folds):
        row_vals = [f"{results[m][i]:.4f}" for m in results.keys()]
        print(f"{i + 1}\t" + "\t".join(row_vals))

    print("\n=== 平均结果 ===")
    for m, vals in results.items():
        print(f"{m}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


# =============== Part 1: 随机数据集 + 嵌套交叉验证 ===============

def nested_cv_svm_random_data(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    random_state=42
):
    print("===== Part 1: 随机二分类数据集上的 SVM 嵌套交叉验证 =====")

    # 1. 生成随机二分类数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state
    )

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # SVM + 标准化
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", SVC())
    ])

    # 超参数搜索空间
    # 注意：gamma 对 linear kernel 没影响，但传进去不会报错，只是忽略
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],  # 主要对 rbf 有用
    }

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    # 2. 外层 10-fold
    fold_idx = 1
    for train_idx, test_idx in outer_cv.split(X, y):
        print(f"\n--- 外层 Fold {fold_idx} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 3. 内层 5-fold + GridSearch
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        print("最佳超参数:", grid.best_params_)
        print(f"内层 CV 最佳平均 Accuracy: {grid.best_score_:.4f}")

        # 4. 使用最佳模型在外层测试集上评估
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"外层测试集: Accuracy={acc:.4f}, "
              f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)

        fold_idx += 1

    # 5. 输出每折以及平均指标
    print_fold_results(metrics)


# =============== Part 2: Adult 数据集读取与预处理 ===============

def load_adult_dataset(
    train_path="adult.data",
    test_path="adult.test"
):
    """
    读取 Adult 数据集 (UCI)，返回 train/test 的 DataFrame 和标签。
    """
    # 列名根据 UCI Adult 数据集文档
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"
    ]

    # adult.data 没有表头
    train_df = pd.read_csv(
        train_path,
        header=None,
        names=columns,
        na_values=" ?",
        skipinitialspace=True
    )

    # adult.test 第一行是表头或说明行，一般需要跳过
    test_df = pd.read_csv(
        test_path,
        header=0,                 # 跳过第一行
        names=columns,
        na_values=" ?",
        skipinitialspace=True
    )

    # 去掉测试集标签末尾的 "."，如 '>50K.' -> '>50K'
    test_df["income"] = test_df["income"].astype(str).str.replace(".", "", regex=False).str.strip()

    # 删除有缺失值的样本（简化处理）
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    X_train = train_df.drop("income", axis=1)
    y_train = train_df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    X_test = test_df.drop("income", axis=1)
    y_test = test_df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    return X_train, y_train, X_test, y_test


def build_preprocessor(
    all_features_df: pd.DataFrame,
    feature_subset: list | None = None
):
    """
    构建 ColumnTransformer：
    - 数值特征：StandardScaler
    - 类别特征：OneHotEncoder(handle_unknown='ignore')

    feature_subset:
        - None: 使用所有特征
        - list: 仅使用给定的列名子集
    """
    if feature_subset is not None:
        df = all_features_df[feature_subset]
    else:
        df = all_features_df

    # 简单区分数值 / 类别特征
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


# =============== Part 2: Adult 数据集上的 SVM 实验 ===============

def svm_on_adult(
    train_path="adult.data",
    test_path="adult.test"
):
    print("\n===== Part 2: Adult 数据集上的 SVM 实验 =====")

    X_train, y_train, X_test, y_test = load_adult_dataset(train_path, test_path)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 共用的参数网格
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- 实验 A: 使用所有特征 ---
    print("\n--- 实验 A: 使用所有特征 ---")
    preprocessor_all, num_all, cat_all = build_preprocessor(X_train, feature_subset=None)
    print("使用的数值特征:", num_all)
    print("使用的类别特征:", cat_all)

    pipe_all = Pipeline(steps=[
        ("preprocess", preprocessor_all),
        ("clf", SVC())
    ])

    grid_all = GridSearchCV(
        estimator=pipe_all,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_all.fit(X_train, y_train)
    print("最佳超参数（所有特征）:", grid_all.best_params_)
    print(f"训练集 5-fold CV 最佳平均 Accuracy（所有特征）: {grid_all.best_score_:.4f}")

    best_all = grid_all.best_estimator_
    y_pred_test_all = best_all.predict(X_test)
    acc_all = accuracy_score(y_test, y_pred_test_all)
    print(f"测试集 Accuracy（所有特征）: {acc_all:.4f}")

    # --- 实验 B: 使用特征子集 ---
    print("\n--- 实验 B: 使用特征子集 ---")
    # 与前面的 Logistic Regression 版本保持一致，手动选择一部分特征
    feature_subset = [
        "age",
        "education-num",
        "hours-per-week",
        "capital-gain",
        "capital-loss",
        "marital-status",
        "occupation",
        "sex"
    ]
    print("特征子集:", feature_subset)

    preprocessor_sub, num_sub, cat_sub = build_preprocessor(X_train, feature_subset=feature_subset)
    print("子集中数值特征:", num_sub)
    print("子集中类别特征:", cat_sub)

    pipe_sub = Pipeline(steps=[
        ("preprocess", preprocessor_sub),
        ("clf", SVC())
    ])

    grid_sub = GridSearchCV(
        estimator=pipe_sub,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_sub.fit(X_train[feature_subset], y_train)
    print("最佳超参数（特征子集）:", grid_sub.best_params_)
    print(f"训练集 5-fold CV 最佳平均 Accuracy（特征子集）: {grid_sub.best_score_:.4f}")

    best_sub = grid_sub.best_estimator_
    y_pred_test_sub = best_sub.predict(X_test[feature_subset])
    acc_sub = accuracy_score(y_test, y_pred_test_sub)
    print(f"测试集 Accuracy（特征子集）: {acc_sub:.4f}")

    # --- 简单比较 ---
    print("\n=== 对比：全部特征 vs 特征子集 ===")
    print(f"测试集 Accuracy（全部特征）: {acc_all:.4f}")
    print(f"测试集 Accuracy（特征子集）: {acc_sub:.4f}")
    if acc_sub > acc_all:
        print("使用特征子集在测试集上表现更好（更高的准确率）。")
    elif acc_sub < acc_all:
        print("使用全部特征在测试集上表现更好。")
    else:
        print("两者在测试集上的准确率相同。")


# =============== main ===============

if __name__ == "__main__":
    # Part 1: 随机数据集 + 嵌套交叉验证
    nested_cv_svm_random_data()

    # Part 2: Adult 数据集 + SVM
    # 需要确保当前目录下存在 adult.data 和 adult.test
    try:
        svm_on_adult("adult.data", "adult.test")
    except FileNotFoundError:
        print("\n未找到 adult.data 或 adult.test，跳过 Part 2（Bonus 部分）。")
