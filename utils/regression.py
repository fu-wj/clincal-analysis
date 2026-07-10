import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import io

# 1. 定义 CNS / SCI 级别期刊标准配色盘
JOURNAL_PALETTES = {
    "Nature / NPG": {"primary": "#E64B35", "error": "#4DBBD5", "grid": "#EFEFEF"},
    "Lancet": {"primary": "#00468B", "error": "#ED0000", "grid": "#F4F4F4"},
    "Science / AAAS": {"primary": "#1A1A1A", "error": "#999999", "grid": "#F0F0F0"},
    "JCO (Clinical Oncology)": {"primary": "#002A54", "error": "#B8860B", "grid": "#FAFAFA"}
}


def run_univariate_regression(df, y_var, x_vars, model_type='Logistic', time_var=None):
    import statsmodels.api as sm
    import numpy as np

    y = df[y_var].astype(float)
    results = []
    for var in x_vars:
        X_var = df[var].copy()
        if X_var.dtype == object:
            X_var = pd.to_numeric(X_var, errors='coerce')
        X = sm.add_constant(X_var.astype(float))
        try:
            if model_type == 'Logistic':
                model = sm.Logit(y, X).fit(disp=0)
                p = model.pvalues[var]
                or_ = np.exp(model.params[var])
                ci = np.exp(model.conf_int().loc[var])
                results.append([var, or_, ci[0], ci[1], p])
            # ... 其他模型类型
        except Exception as e:
            results.append([var, np.nan, np.nan, np.nan, np.nan])
    res_df = pd.DataFrame(results, columns=['Variable', 'Odds Ratio (OR)', 'Lower_95_CI', 'Upper_95_CI', 'P_value'])
    return res_df


def run_multivariate_regression(df, y_var, x_vars, model_type='Logistic', time_var=None):
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np

    # 确保 X 和 y 是数值类型，并处理缺失值（虽然应该已经填补）
    X = df[x_vars].copy()
    y = df[y_var].copy()

    # 将可能存在的 object 列转为数值
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                pass
    # 强制转换整个 X 为 float
    X = X.astype(float)
    y = y.astype(float)

    # 添加常数项
    X = sm.add_constant(X)

    if model_type == 'Logistic':
        model = sm.Logit(y, X).fit(disp=0)
    else:
        # Cox 略
        pass

    params = model.params.drop('const', errors='ignore')
    conf = model.conf_int().drop('const', errors='ignore')
    pvalues = model.pvalues.drop('const', errors='ignore')

    res = pd.DataFrame({
        'Variable': params.index,
        'Odds Ratio (OR)': np.exp(params.values),
        'Lower_95_CI': np.exp(conf.iloc[:, 0].values),
        'Upper_95_CI': np.exp(conf.iloc[:, 1].values),
        'P_value': pvalues.values
    })
    return res


def plot_forest_chart(results_df, metric_col=None, ci_low_col=None, ci_high_col=None, palette_name=None, title=None):
    """
    通用森林图，自动识别效应量和置信区间的列名。
    results_df 需包含 'Variable' 列或以其 index 作为变量名。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.plot_style import get_palette

    # 确保有变量名列
    if 'Variable' not in results_df.columns:
        results_df = results_df.reset_index()
        if 'index' in results_df.columns:
            results_df.rename(columns={'index': 'Variable'}, inplace=True)
        else:
            raise KeyError("DataFrame must have 'Variable' column or index.")

    # 自动探测效应量列
    possible_metrics = ['Odds Ratio (OR)', 'Hazard Ratio (HR)', 'OR', 'HR', 'exp(coef)', 'Coef.']
    if metric_col is None or metric_col not in results_df.columns:
        for col in possible_metrics:
            if col in results_df.columns:
                metric_col = col
                break
        else:
            raise KeyError(f"No metric column found. Available: {results_df.columns.tolist()}")

    # 自动探测置信区间列
    lower_candidates = ['Lower_95_CI', 'CI_lower', 'exp(coef) lower 95%', '2.5%', 'Lower']
    upper_candidates = ['Upper_95_CI', 'CI_upper', 'exp(coef) upper 95%', '97.5%', 'Upper']

    if ci_low_col is None or ci_low_col not in results_df.columns:
        for col in lower_candidates:
            if col in results_df.columns:
                ci_low_col = col
                break
        else:
            raise KeyError(f"Lower CI column not found. Available: {results_df.columns.tolist()}")

    if ci_high_col is None or ci_high_col not in results_df.columns:
        for col in upper_candidates:
            if col in results_df.columns:
                ci_high_col = col
                break
        else:
            raise KeyError(f"Upper CI column not found.")

    # 绘制
    y_pos = range(len(results_df))
    means = results_df[metric_col].astype(float)
    left_err = means - results_df[ci_low_col].astype(float)
    right_err = results_df[ci_high_col].astype(float) - means

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(results_df))))
    ax.errorbar(means, y_pos, xerr=[left_err, right_err], fmt='o', capsize=5, color='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['Variable'])
    ax.axvline(1, color='gray', linestyle='--')
    ax.set_xlabel(metric_col)
    ax.set_title(title or f'Forest Plot ({metric_col})')
    plt.tight_layout()
    return fig
