import pandas as pd
from tableone import TableOne


def generate_tableone(df, groupby_var, categorical_vars, nonnormal_vars):
    """
    利用 tableone 库自动生成基线表，包含 P 值计算。
    """
    columns = df.columns.tolist()

    # 设定分组变量
    groupby = groupby_var if groupby_var != "无分组" else None
    if groupby:
        columns.remove(groupby)

    mytable = TableOne(
        df,
        columns=columns,
        categorical=categorical_vars,
        groupby=groupby,
        nonnormal=nonnormal_vars,
        pval=True if groupby else False,
        htest_name=True  # 显示使用的统计检验方法(如 T-test, Chi-squared)
    )
    return mytable