import re
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗列名：将所有中文字符和 Python 不支持的特殊符号转化为 '_'。
    """
    new_cols = []
    for col in df.columns:
        # 1. 替换中文和非字母数字的特殊符号为下划线
        cleaned = re.sub(r'[^\w\s]', '_', str(col))
        cleaned = re.sub(r'[\u4e00-\u9fa5]', '_', cleaned)
        # 2. 替换空格为下划线
        cleaned = re.sub(r'\s+', '_', cleaned)
        # 3. 避免连续下划线
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        new_cols.append(cleaned)

    df.columns = new_cols
    return df