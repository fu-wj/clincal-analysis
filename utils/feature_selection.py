import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

# ---------- 原有函数 ----------
def run_lasso_selection(X, y):
    lasso = LogisticRegressionCV(
        Cs=5, penalty='l1', solver='liblinear',
        random_state=42, max_iter=10000
    )
    lasso.fit(X, y)
    coefs = lasso.coef_[0]
    lasso_features = X.columns[coefs != 0].tolist()
    importance_df = pd.DataFrame({
        'Variable': X.columns,
        'Lasso_Coef': coefs
    }).sort_values(by='Lasso_Coef', key=abs, ascending=False)
    return lasso_features, importance_df

def run_boruta_selection(X, y):
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
    feat_selector = BorutaPy(
        rf, n_estimators='auto',
        verbose=0, random_state=42,
        max_iter=100
    )
    feat_selector.fit(X.values, y.values)
    boruta_features = X.columns[feat_selector.support_].tolist()
    tentative_features = X.columns[feat_selector.support_weak_].tolist()
    rf.fit(X, y)
    importance_df = pd.DataFrame({
        'Variable': X.columns,
        'RF_Importance': rf.feature_importances_,
        'Status': ['Confirmed' if s else ('Tentative' if w else 'Rejected')
                   for s, w in zip(feat_selector.support_, feat_selector.support_weak_)]
    }).sort_values(by='RF_Importance', ascending=False)
    return boruta_features, tentative_features, importance_df

def plot_feature_comparison(lasso_df, boruta_df, selected_strategy, intersection_list, palette_colors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    # LASSO
    top_lasso = lasso_df.head(10)[::-1]
    ax1.barh(top_lasso['Variable'], top_lasso['Lasso_Coef'],
             color=palette_colors["primary"], edgecolor='black', height=0.6)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel("LASSO Coefficient", fontsize=10, fontweight='bold')
    ax1.set_title("Top 10 LASSO Selected Features", fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    # Boruta
    top_boruta = boruta_df.head(10)[::-1]
    colors = [palette_colors["primary"] if s == 'Confirmed' else
              (palette_colors.get("error", "#4DBBD5") if s == 'Tentative' else '#D3D3D3')
              for s in top_boruta['Status']]
    ax2.barh(top_boruta['Variable'], top_boruta['RF_Importance'],
             color=colors, edgecolor='black', height=0.6)
    ax2.set_xlabel("Random Forest Gini Importance", fontsize=10, fontweight='bold')
    ax2.set_title("Top 10 Boruta Features & Status", fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# ---------- 新增绘图函数 ----------
def plot_lasso_path(X, y, alphas=50, palette_colors=None):
    """
    绘制 LASSO 系数路径图（二分类 Logistic 回归）
    对多个 C 值拟合模型，记录系数变化。
    """
    if palette_colors is None:
        palette_colors = {"primary": "#E64B35"}
    c_vals = np.logspace(-4, 2, alphas)
    coefs = []
    for c in c_vals:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=c, random_state=42, max_iter=10000)
        clf.fit(X, y)
        coefs.append(clf.coef_[0])
    coefs = np.array(coefs)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for i in range(coefs.shape[1]):
        ax.plot(np.log10(c_vals), coefs[:, i], label=X.columns[i], lw=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('log10(C)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
    ax.set_title('LASSO Coefficient Path (Logistic)', fontsize=11, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_boruta_importance_distribution(boruta_imp, palette_colors=None):
    """
    绘制 Boruta 特征重要性在三种状态下的分布（箱线图）
    注意：兼容 Matplotlib 新版本，不使用 'labels' 参数。
    """
    if palette_colors is None:
        palette_colors = {"primary": "#E64B35", "error": "#4DBBD5", "secondary": "#D3D3D3"}
    status_order = ['Confirmed', 'Tentative', 'Rejected']
    data = []
    for status in status_order:
        vals = boruta_imp[boruta_imp['Status'] == status]['RF_Importance'].values
        if len(vals) > 0:
            data.append(vals)
        else:
            data.append([0])  # 占位，避免空数据
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    # 绘制箱线图，不传入 labels
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(edgecolor='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    medianprops=dict(color='red'))
    # 设置横坐标标签
    ax.set_xticklabels(status_order)
    # 填充颜色
    colors = [palette_colors.get("primary", "#E64B35"),
              palette_colors.get("error", "#4DBBD5"),
              palette_colors.get("secondary", "#D3D3D3")]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Random Forest Importance', fontsize=10, fontweight='bold')
    ax.set_title('Boruta Feature Importance Distribution', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_venn_diagram(lasso_feats, boruta_feats, tentative_feats=None, palette_colors=None):
    """
    绘制 Venn 图展示 LASSO 和 Boruta 选择的重叠情况
    """
    if palette_colors is None:
        palette_colors = {"primary": "#E64B35", "error": "#4DBBD5"}
    set_l = set(lasso_feats)
    set_b = set(boruta_feats)
    if tentative_feats:
        set_b = set_b | set(tentative_feats)  # 将暂定也视为 Boruta 阳性（可选）
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    v = venn2([set_l, set_b], set_labels=('LASSO', 'Boruta'),
              set_colors=(palette_colors.get("primary", "#E64B35"),
                          palette_colors.get("error", "#4DBBD5")),
              alpha=0.6)
    for subset in ('10', '01', '11'):
        label = v.get_label_by_id(subset)
        if label:
            label.set_fontsize(12)
            label.set_weight('bold')
    ax.set_title('LASSO ∩ Boruta Selected Features', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(X, features, palette_colors=None):
    """
    绘制选定特征间的 Spearman 相关系数热图
    """
    if palette_colors is None:
        palette_colors = {"primary": "#E64B35"}
    sub_X = X[features]
    corr = sub_X.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cmap = sns.diverging_palette(250, 30, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Spearman Correlation Heatmap (Selected Features)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig