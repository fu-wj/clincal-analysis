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


def run_univariate_regression(df, y_var, x_vars, model_type="Logistic", time_var=None):
    """
    批量运行单因素回归分析
    """
    results = []

    for x in x_vars:
        try:
            if model_type == "Logistic":
                # 显式转换表述，确保稳定性
                formula = f"{y_var} ~ {x}"
                model = smf.logit(formula, data=df).fit(disp=0)

                param = model.params[x]
                conf = model.conf_int().loc[x]
                pvalue = model.pvalues[x]

                metric_name = "Odds Ratio (OR)"
                value = np.exp(param)
                lower_ci = np.exp(conf[0])
                upper_ci = np.exp(conf[1])

            elif model_type == "Cox" and time_var:
                cph = CoxPHFitter()
                # 仅筛选当前所需的生存三要素数据，剔除缺失值
                analysis_df = df[[time_var, y_var, x]].dropna()
                cph.fit(analysis_df, duration_col=time_var, event_col=y_var)

                summary = cph.summary.loc[x]
                metric_name = "Hazard Ratio (HR)"
                value = summary['exp(coef)']
                lower_ci = summary['exp(coef) lower 95%']
                upper_ci = summary['exp(coef) upper 95%']
                pvalue = summary['p']

            results.append({
                "Variable": x,
                f"{metric_name}": value,
                "Lower_95_CI": lower_ci,
                "Upper_95_CI": upper_ci,
                "P_value": pvalue
            })
        except Exception as e:
            continue

    return pd.DataFrame(results)


def run_multivariate_regression(df, y_var, x_vars, model_type="Logistic", time_var=None):
    """
    运行多因素回归分析
    """
    if not x_vars:
        return pd.DataFrame()

    try:
        if model_type == "Logistic":
            formula = f"{y_var} ~ " + " + ".join(x_vars)
            model = smf.logit(formula, data=df).fit(disp=0)

            results = []
            for x in x_vars:
                results.append({
                    "Variable": x,
                    "Odds Ratio (OR)": np.exp(model.params[x]),
                    "Lower_95_CI": np.exp(model.conf_int().loc[x][0]),
                    "Upper_95_CI": np.exp(model.conf_int().loc[x][1]),
                    "P_value": model.pvalues[x]
                })
            return pd.DataFrame(results)

        elif model_type == "Cox" and time_var:
            cph = CoxPHFitter()
            analysis_df = df[[time_var, y_var] + x_vars].dropna()
            cph.fit(analysis_df, duration_col=time_var, event_col=y_var)

            results = []
            for x in x_vars:
                summary = cph.summary.loc[x]
                results.append({
                    "Variable": x,
                    "Hazard Ratio (HR)": summary['exp(coef)'],
                    "Lower_95_CI": summary['exp(coef) lower 95%'],
                    "Upper_95_CI": summary['exp(coef) upper 95%'],
                    "P_value": summary['p']
                })
            return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def plot_forest_chart(res_df, metric_col, palette_name="Nature / NPG"):
    """
    绘制符合 SCI/CNS 顶刊级别的英文森林图
    """
    palette = JOURNAL_PALETTES.get(palette_name, JOURNAL_PALETTES["Nature / NPG"])

    # 按照变量顺序倒序排列，保证图表从上往下契合表格顺序
    plot_df = res_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7, max(3, len(plot_df) * 0.45)), dpi=300)

    # 绘制背景参考网格线
    ax.xaxis.grid(True, linestyle='--', alpha=0.6, color=palette["grid"])
    ax.set_axisbelow(True)

    # 绘制无效线 (OR/HR = 1)
    ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

    # 绘制置信区间与中心点
    for i, row in plot_df.iterrows():
        # 计算误差线的左右长度
        left_err = row[metric_col] - row['Lower_95_CI']
        right_err = row['Upper_95_CI'] - row[metric_col]

        ax.errorbar(
            x=row[metric_col], y=y_pos[i],
            xerr=[[left_err], [right_err]],
            fmt='o', color=palette["primary"],
            ecolor=palette["error"], elinewidth=1.5,
            capsize=4, markersize=7, mec='black', mew=0.5
        )

        # 在图表右侧边缘标注 P 值和具体数值 (英文学术规范)
        p_text = f"P={row['P_value']:.3f}" if row['P_value'] >= 0.001 else "P<0.001"
        anno_text = f"{row[metric_col]:.2f} ({row['Lower_95_CI']:.2f}-{row['Upper_95_CI']:.2f})  {p_text}"
        ax.text(ax.get_xlim()[1] * 1.05, y_pos[i], anno_text, va='center', fontsize=9, fontfamily='sans-serif')

    # 图表细节精修
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Variable'], fontsize=10, fontfamily='sans-serif')
    ax.set_xlabel(metric_col, fontsize=11, fontweight='bold', fontfamily='sans-serif')

    # 移除顶部和右侧的边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig