import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

from utils.plot_style import get_palette, PALETTES
import matplotlib.pyplot as plt
def get_explainer_and_values(pipeline, X_train, m_name):
    """
    剥离 Pipeline，自适应选择 TreeExplainer 或通用 Explainer，并计算 SHAP 值。
    """
    scaler = pipeline.steps[0][1]
    model = pipeline.steps[1][1]

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

    tree_models = ['Random Forest', 'XGBoost', 'Decision Tree', 'Gradient Boosting']
    is_tree = m_name in tree_models

    if is_tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train_scaled)

        try:
            interaction_values = explainer.shap_interaction_values(X_train_scaled)
        except:
            interaction_values = None
    else:
        background = shap.kmeans(X_train_scaled, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values_array = explainer.shap_values(X_train_scaled)
        shap_values = shap.Explanation(values=shap_values_array,
                                       base_values=np.repeat(explainer.expected_value, len(X_train_scaled)),
                                       data=X_train_scaled,
                                       feature_names=X_train.columns.tolist())
        interaction_values = None

    return explainer, shap_values, interaction_values, X_train_scaled


plt.rcParams['font.family'] = 'Times New Roman'
def create_figure():
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.clf()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_shap_bar(shap_values):
    fig = create_figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_shap_beeswarm(shap_values):
    fig = create_figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_shap_heatmap(shap_values):
    fig = plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.heatmap(shap_values, show=False)
    plt.tight_layout()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_shap_waterfall(shap_values, instance_index=0):
    fig = create_figure()
    shap.plots.waterfall(shap_values[instance_index], show=False)
    plt.tight_layout()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_shap_force(explainer, shap_values, instance_index=0):
    fig = plt.figure(figsize=(12, 3), dpi=300)
    if hasattr(shap_values[instance_index], 'base_values'):
        base_val = shap_values[instance_index].base_values
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0] if len(base_val) > 0 else 0
        val = shap_values[instance_index].values
        data = shap_values[instance_index].data
    else:
        base_val = explainer.expected_value
        val = shap_values[instance_index]
        data = shap_values.data[instance_index]

    # 确保 data 是 numpy 数组
    if hasattr(data, 'values'):
        data = data.values
    shap.force_plot(base_val, val, data,
                    feature_names=shap_values.feature_names,
                    matplotlib=True, show=False, figsize=(12, 3))
    plt.tight_layout()
    return plt.gcf()


plt.rcParams['font.family'] = 'Times New Roman'
def plot_shap_dependence(shap_values, feature_name, palette_name="Nature"):
    palette = get_palette(palette_name)
    """
    手动绘制 SHAP 依赖图，避免 shap.plots.scatter 的内部数据类型错误。
    """
    # 提取特征值 (x轴)
    if hasattr(shap_values.data, 'values'):
        x = shap_values.data[feature_name].values
    else:
        x = shap_values.data[feature_name]
    # 提取该特征的 SHAP 值 (y轴)
    y = shap_values[:, feature_name].values

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(x, y, c=palette["secondary"], edgecolors='black', alpha=0.7, s=60)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(feature_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('SHAP value for ' + feature_name, fontsize=12, fontweight='bold')
    ax.set_title(f'SHAP Dependence Plot: {feature_name}', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_interaction_matrix(interaction_values, feature_names):
    """交互作用绝对值的热力图矩阵"""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    mean_abs_interaction = np.abs(interaction_values).mean(0)
    sns.heatmap(mean_abs_interaction, xticklabels=feature_names, yticklabels=feature_names,
                cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Mean Absolute SHAP Interaction Matrix", fontweight="bold")
    plt.tight_layout()
    return fig


plt.rcParams['font.family'] = 'Times New Roman'
def plot_interaction_network(interaction_values, feature_names, threshold=0.01):
    """交互作用强度网络图"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    mean_abs_interaction = np.abs(interaction_values).mean(0)

    G = nx.Graph()
    for i, name in enumerate(feature_names):
        G.add_node(name)

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            weight = mean_abs_interaction[i, j]
            if weight > threshold:
                G.add_edge(feature_names[i], feature_names[j], weight=weight)

    if len(G.edges) == 0:
        ax.text(0.5, 0.5, "No strong interactions found above threshold.", ha='center', va='center')
        ax.axis('off')
        return fig

    edges = G.edges()
    weights = [G[u][v]['weight'] * 100 for u, v in edges]

    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='#4DBBD5', node_size=1500, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif", font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='#E64B35', alpha=0.6, ax=ax)

    ax.set_title("Feature Interaction Network", fontweight="bold")
    ax.axis('off')
    return fig