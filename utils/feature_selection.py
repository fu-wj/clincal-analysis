import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegression   # 添加 LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

def run_lasso_selection(X, y, cv=5, alpha_strategy='lambda.min', random_state=0):
    if X.isnull().sum().sum() > 0:
        X = X.fillna(0)
    if y.isnull().sum() > 0:
        y = y.fillna(y.mode()[0] if not y.mode().empty else 0)

    lasso_cv = LassoCV(cv=cv, random_state=random_state, max_iter=5000).fit(X, y)

    mse_path = lasso_cv.mse_path_
    mean_mse = mse_path.mean(axis=1)
    std_mse = mse_path.std(axis=1)
    alphas = lasso_cv.alphas_

    if alpha_strategy == 'lambda.min':
        best_alpha = lasso_cv.alpha_
    elif alpha_strategy == 'lambda.1se':
        idx_min = np.argmin(mean_mse)
        min_mse = mean_mse[idx_min]
        thres = min_mse + std_mse[idx_min]
        best_alpha = alphas[idx_min]
        for i in range(idx_min, -1, -1):
            if mean_mse[i] <= thres:
                best_alpha = alphas[i]
                break
    else:
        raise ValueError("alpha_strategy must be 'lambda.min' or 'lambda.1se'")

    if best_alpha == lasso_cv.alpha_:
        coef = lasso_cv.coef_
    else:
        idx = np.abs(alphas - best_alpha).argmin()
        coef = lasso_cv.coef_path_[:, idx]

    feature_names = X.columns.tolist()
    selected_features = [feature_names[i] for i, c in enumerate(coef) if abs(c) > 1e-5]
    importance_df = pd.DataFrame({
        'Variable': feature_names,
        'Coefficient': coef
    }).sort_values(by='Coefficient', ascending=False)

    return selected_features, importance_df

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
        'Importance': rf.feature_importances_,
        'Status': ['Confirmed' if s else ('Tentative' if w else 'Rejected')
                   for s, w in zip(feat_selector.support_, feat_selector.support_weak_)]
    }).sort_values(by='Importance', ascending=False)
    return boruta_features, tentative_features, importance_df

def plot_feature_comparison(lasso_imp, boruta_imp, strategy, both_positive, palette_colors):
    # 处理 palette_colors 可能是列表的情况
    if isinstance(palette_colors, list):
        colors = palette_colors
    else:
        colors = [palette_colors.get("primary", "#E64B35"), palette_colors.get("error", "#4DBBD5")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    top_lasso = lasso_imp.sort_values(by='Coefficient', ascending=True).tail(15)
    ax1.barh(top_lasso['Variable'], top_lasso['Coefficient'], color=colors[0])
    ax1.set_title('LASSO Coefficients', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)

    importance_col = 'Importance'  # 列名已统一为 Importance
    top_boruta = boruta_imp.sort_values(by=importance_col, ascending=True).tail(15)
    ax2.barh(top_boruta['Variable'], top_boruta[importance_col], color=colors[1 % len(colors)])
    ax2.set_title('Boruta Feature Importance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Importance')

    plt.tight_layout()
    return fig

def plot_lasso_path(X, y, alphas=50, palette_colors=None):
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
    # 颜色处理（兼容列表和字典）
    if isinstance(palette_colors, list):
        colors = palette_colors[:3] if len(palette_colors) >= 3 else palette_colors * 3
    else:
        colors = [
            palette_colors.get("primary", "#E64B35"),
            palette_colors.get("error", "#4DBBD5"),
            palette_colors.get("secondary", "#D3D3D3")
        ]

    status_order = ['Confirmed', 'Tentative', 'Rejected']
    data = []
    for status in status_order:
        vals = boruta_imp[boruta_imp['Status'] == status]['Importance'].values  # 列名已改为 Importance
        if len(vals) > 0:
            data.append(vals)
        else:
            data.append([0])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(edgecolor='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    medianprops=dict(color='red'))
    ax.set_xticklabels(status_order)
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
    # 颜色处理
    if isinstance(palette_colors, list):
        col1, col2 = palette_colors[0], palette_colors[1 % len(palette_colors)]
    else:
        col1 = palette_colors.get("primary", "#E64B35")
        col2 = palette_colors.get("error", "#4DBBD5")

    set_l = set(lasso_feats)
    set_b = set(boruta_feats)
    if tentative_feats:
        set_b = set_b | set(tentative_feats)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    v = venn2([set_l, set_b], set_labels=('LASSO', 'Boruta'),
              set_colors=(col1, col2), alpha=0.6)
    for subset in ('10', '01', '11'):
        label = v.get_label_by_id(subset)
        if label:
            label.set_fontsize(12)
            label.set_weight('bold')
    ax.set_title('LASSO ∩ Boruta Selected Features', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(X, features, palette_colors=None):
    sub_X = X[features]
    corr = sub_X.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cmap = sns.diverging_palette(250, 30, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Spearman Correlation Heatmap (Selected Features)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig
