import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils.plot_style import get_palette, PALETTES


# ================= 1. 模型字典 =================
def get_model_dict():
    """返回所有监督学习模型的流水线（均含标准化）"""
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "XGBoost": make_pipeline(StandardScaler(),
                                 XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        "Neural Network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, random_state=42))
    }


# ================= 2. 高级临床指标计算 =================
def calculate_auc_ci_and_cutoff(y_true, y_prob, n_bootstraps=1000):
    """
    计算 AUC 的 95% 置信区间 (Bootstrap) 及最佳截断值 (Youden)
    返回: best_cutoff, sensitivity, specificity, lower_ci, upper_ci
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_idx = tpr - fpr
    best_idx = np.argmax(youden_idx)
    best_cutoff = thresholds[best_idx]
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]

    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    if len(sorted_scores) == 0:
        return best_cutoff, sensitivity, specificity, 0, 0

    lower_ci = sorted_scores[int(0.025 * len(sorted_scores))]
    upper_ci = sorted_scores[int(0.975 * len(sorted_scores))]

    return best_cutoff, sensitivity, specificity, lower_ci, upper_ci


def evaluate_predictions(y_true, y_pred, y_prob, model_name):
    """计算所有评价指标并封装为字典"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    best_cutoff, sens, spec, ci_low, ci_high = calculate_auc_ci_and_cutoff(y_true, y_prob)
    return {
        "Model": model_name,
        "Accuracy": acc,
        "F1_Score": f1,
        "AUC": auc,
        "AUC_95%_CI": f"{ci_low:.3f} - {ci_high:.3f}",
        "Optimal_Cutoff": best_cutoff,
        "Sensitivity": sens,
        "Specificity": spec
    }


# ================= 3. 单变量 AUC 计算 =================
def calculate_univariate_auc(X, y, feature_names=None):
    """
    对每个特征单独计算其预测 y 的 AUC 值（直接使用特征值作为预测得分）
    返回: DataFrame with columns: Feature, AUC
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    auc_list = []
    for feat in feature_names:
        try:
            auc = roc_auc_score(y, X[feat])
        except Exception:
            auc = np.nan
        auc_list.append({"Feature": feat, "AUC": auc})
    df = pd.DataFrame(auc_list).sort_values(by="AUC", ascending=False)
    return df


# ================= 4. 绘图函数 =================
def plot_multi_roc(y_true, y_probs_dict, title_prefix="Validation", palette_name="Nature"):
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    colors = palette["colors"]  # 使用10色列表
    for idx, (model_name, y_prob) in enumerate(y_probs_dict.items()):
        color = colors[idx % len(colors)]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_val:.3f})', lw=2, color=color, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix} Set ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def plot_metric_bars(metrics_df, metric_col="AUC", title_prefix="Validation", palette_name="Nature"):
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sorted_df = metrics_df.sort_values(by=metric_col, ascending=False)

    # 检查是否有 AUC_95%_CI 列，且 metric_col 为 AUC
    has_ci = "AUC_95%_CI" in metrics_df.columns and metric_col == "AUC"
    if has_ci:
        cis = []
        for ci_str in sorted_df["AUC_95%_CI"]:
            try:
                low, high = map(float, ci_str.split(" - "))
            except:
                low, high = np.nan, np.nan
            cis.append((low, high))
        means = sorted_df[metric_col].values
        yerr_low = [means[i] - cis[i][0] if not np.isnan(cis[i][0]) else 0 for i in range(len(cis))]
        yerr_high = [cis[i][1] - means[i] if not np.isnan(cis[i][1]) else 0 for i in range(len(cis))]
        yerr = [yerr_low, yerr_high]
    else:
        yerr = None

    bars = ax.bar(sorted_df['Model'], sorted_df[metric_col],
                  color=palette["secondary"], edgecolor='black', width=0.6,
                  yerr=yerr, capsize=5, error_kw={'ecolor': 'black', 'elinewidth': 1.5})

    ax.set_ylim(0, 1.05 if metric_col in ["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"] else None)
    ax.set_ylabel(metric_col, fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric_col} ({title_prefix} Set)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(y_true, y_preds_dict, title_prefix=""):
    n_models = len(y_preds_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), dpi=300)
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, (model_name, y_pred) in enumerate(y_preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    annot_kws={"size": 12, "weight": "bold"})
        # 如果有 title_prefix，则添加到标题前面
        title = f"{title_prefix} - {model_name}" if title_prefix else model_name
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=9)
        axes[i].set_ylabel('True Label', fontsize=9)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def plot_univariate_auc_bar(df_auc, title="Univariate AUC of Selected Features", palette_name="Nature"):
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    df_sorted = df_auc.sort_values(by="AUC", ascending=True)
    bars = ax.barh(df_sorted["Feature"], df_sorted["AUC"],
                   color=palette["primary"], edgecolor='black', height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC (Area Under ROC Curve)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.3f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_univariate_auc_forest(df_auc, title="Univariate AUC Forest Plot", palette_name="Nature"):
    """单变量 AUC 森林图（带误差条，但由于单变量 AUC 没有 CI，仅展示点估计）"""
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    df_sorted = df_auc.sort_values(by="AUC", ascending=True)
    y_pos = np.arange(len(df_sorted))
    means = df_sorted["AUC"].values
    # 由于无 CI，只画点
    ax.scatter(means, y_pos, color=palette["primary"], s=80, edgecolors='black', zorder=5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["Feature"])
    ax.set_xlabel("AUC", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ================= 5. 新增图表类型 =================
def plot_forest_style(metrics_df, metric_col="AUC", title_prefix="Validation", palette_name="Nature"):
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 解析 CI
    cis = []
    for ci_str in metrics_df["AUC_95%_CI"]:
        try:
            low, high = map(float, ci_str.split(" - "))
        except:
            low, high = np.nan, np.nan
        cis.append((low, high))

    # 排序
    sorted_df = metrics_df.sort_values(by=metric_col, ascending=True)
    y_pos = np.arange(len(sorted_df))
    means = sorted_df[metric_col].values

    # 计算误差，确保非负
    low_err = np.maximum(0, means - np.array([c[0] for c in cis]))
    high_err = np.maximum(0, np.array([c[1] for c in cis]) - means)

    ax.errorbar(means, y_pos, xerr=[low_err, high_err], fmt='o',
                color=palette["primary"], ecolor=palette["secondary"],
                capsize=5, elinewidth=2, markersize=10)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['Model'])
    ax.set_xlabel(metric_col, fontsize=12, fontweight='bold')
    ax.set_title(f'Forest Plot: {metric_col} with 95% CI ({title_prefix})', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_scatter_comparison(metrics_df, x_metric="AUC", y_metric="F1_Score", title_prefix="Validation",
                            palette_name="Nature"):
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    x = metrics_df[x_metric]
    y = metrics_df[y_metric]
    ax.scatter(x, y, color=palette["primary"], s=100, edgecolors='black', alpha=0.8)
    for i, model in enumerate(metrics_df['Model']):
        ax.annotate(model, (x.iloc[i], y.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel(x_metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{x_metric} vs {y_metric} ({title_prefix})', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_radar_chart(metrics_df, metric_cols=["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"],
                     title_prefix="Validation", palette_name="Nature"):
    palette = get_palette(palette_name)
    df_radar = metrics_df[metric_cols].copy()
    labels = metric_cols
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    colors = palette["colors"]
    for idx, (i, row) in enumerate(metrics_df.iterrows()):
        values = row[metric_cols].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid',
                label=row['Model'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Radar Chart: Multi-Metrics Comparison ({title_prefix})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    plt.tight_layout()
    return fig


def plot_univariate_roc_curves(X, y, feature_names=None, title="Univariate ROC Curves",
                               max_features=10, palette_name="Nature"):
    """
    绘制每个特征的 ROC 曲线（每个特征单独作为预测器）
    参数:
        X: DataFrame，包含特征
        y: 目标变量
        feature_names: 要绘制的特征列表（若为 None，则使用全部）
        max_features: 最多显示的特征数（避免图太拥挤）
        palette_name: 调色板名称
    返回: matplotlib 图形
    """
    if feature_names is None:
        feature_names = X.columns.tolist()

    # 如果特征太多，按 AUC 排序后取前 max_features 个
    from sklearn.metrics import roc_auc_score, roc_curve
    auc_values = {}
    for feat in feature_names:
        try:
            auc = roc_auc_score(y, X[feat])
        except:
            auc = np.nan
        auc_values[feat] = auc
    sorted_features = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, auc in sorted_features if not np.isnan(auc)][:max_features]

    palette = get_palette(palette_name)
    colors = palette["colors"]  # 10种颜色

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for idx, feat in enumerate(top_features):
        fpr, tpr, _ = roc_curve(y, X[feat])
        auc_val = auc_values[feat]
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f'{feat} (AUC={auc_val:.3f})', lw=2, color=color, alpha=0.8)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ================= 决策曲线 (Decision Curve) =================
def plot_decision_curve(y_true, y_prob, model_name, title_prefix="Validation", palette_name="Nature"):
    """
    绘制单个模型的决策曲线
    """
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 生成阈值范围（0 到 1）
    thresholds = np.linspace(0, 1, 100)
    net_benefit = []
    for thresh in thresholds:
        # 预测为正类的概率 >= thresh 则分类为正
        y_pred = (y_prob >= thresh).astype(int)
        # 混淆矩阵
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        n = len(y_true)
        # 净收益 = TP/n - (FP/n) * (thresh/(1-thresh))
        if 1 - thresh == 0:
            nb = tp / n
        else:
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit.append(nb)

    # 绘制 "Treat All" 和 "Treat None" 参考线
    prevalence = np.mean(y_true)
    ax.plot(thresholds, net_benefit, label=f'{model_name}', lw=2, color=palette["primary"])
    ax.plot(thresholds, [prevalence - (1 - prevalence) * (thresh / (1 - thresh)) for thresh in thresholds],
            'k--', lw=1.5, label='Treat All')
    ax.axhline(0, color='gray', linestyle='--', lw=1.5, label='Treat None')

    ax.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax.set_title(f'Decision Curve: {model_name} ({title_prefix})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_calibration_curve(y_true, y_prob, model_name, title_prefix="Validation", palette_name="Nature", n_bins=10):
    """
    绘制校准曲线
    """
    palette = get_palette(palette_name)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 分箱
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=True)
    mean_pred = []
    mean_obs = []
    for i in range(1, n_bins + 1):
        idx = bin_indices == i
        if np.sum(idx) > 0:
            mean_pred.append(np.mean(y_prob[idx]))
            mean_obs.append(np.mean(y_true[idx]))

    ax.plot(mean_pred, mean_obs, 'o-', color=palette["primary"], lw=2, label=model_name)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Fraction of Positive Cases', fontsize=12, fontweight='bold')
    ax.set_title(f'Calibration Curve: {model_name} ({title_prefix})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig
