import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import zipfile
from scipy import stats

# ==================== 密码验证 ====================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True
    password = st.text_input("请输入访问密码", type="password")
    if st.button("登录"):
        if password == st.secrets.get("ACCESS_PASSWORD", "admin"):
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("😕 密码错误，请重试。")
    return False

if not check_password():
    st.stop()

# ==================== 工具函数 ====================
def clean_column_name(col):
    col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
    if col and col[0].isdigit():
        col = '_' + col
    return col

def clean_cell(val):
    if isinstance(val, str):
        val = val.encode('ascii', 'ignore').decode('ascii')
        val = re.sub(r'\s+', '_', val)
        return val if val else 'Unknown'
    return val

PALETTES = {
    "Nature": ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"],
    "Science": ["#394A96","#E51E25","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628","#F781BF","#999999"],
}
def get_palette(name):
    return PALETTES.get(name, PALETTES["Nature"])

# ==================== 页面配置 ====================
st.set_page_config(page_title="临床多组学自动化分析平台", layout="wide")

if 'data_uploaded' not in st.session_state:
    st.session_state.update({
        'data_uploaded': False,
        'step1_done': False,
        'step2_done': False,
        'step3_done': False,
        'df_clean': None,
        'all_outputs': [],
        'final_ml_features': [],
        'complete_df': None,
        'y_selector': None,
        'palette_choice': 'Nature',
        'cat_cols': [],
        'baseline_df': None,
    })

def store_figure(fig, filename):
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    st.session_state.all_outputs = [(f,d) for f,d in st.session_state.all_outputs if not f.startswith(base)]
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    st.session_state.all_outputs.append((f"{base}.pdf", buf.getvalue()))
    buf = io.BytesIO()
    fig.savefig(buf, format='tiff', dpi=300, bbox_inches='tight')
    st.session_state.all_outputs.append((f"{base}.tiff", buf.getvalue()))

def store_dataframe(df, filename):
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    st.session_state.all_outputs = [(f,d) for f,d in st.session_state.all_outputs if not f.startswith(base)]
    st.session_state.all_outputs.append((f"{base}.csv", df.to_csv(index=False).encode('utf-8')))

def store_string(text, filename):
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    st.session_state.all_outputs = [(f,d) for f,d in st.session_state.all_outputs if not f.startswith(base)]
    st.session_state.all_outputs.append((f"{base}.txt", text.encode('utf-8')))

st.markdown("""
    <style>
    .stApp { background-color: #fdfbf7; }
    h1, h2, h3 { color: #4a4266; font-family: "Microsoft YaHei", sans-serif; }
    .stButton>button { background-color: #ff461f; color: white; border-radius: 5px; border: none; }
    .stButton>button:hover { background-color: #c3270b; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 临床数据智能分析与机器学习平台")
st.markdown("一键完成：**基线表 → 回归迭代 → 筛选特征 → 机器学习 → SHAP解释**。支持 CNS 级别可视化下载。")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 分析流程导航")
    mode = st.radio("选择模式", ["主队列完整分析"], key="mode_radio")
    if mode == "主队列完整分析":
        st.markdown("---")
        for i in range(1, 7):
            key = f'step{i}_done'
            labels = ["数据预处理与基线表", "单/多因素回归与迭代", "LASSO/Boruta 特征筛选",
                      "机器学习建模与评估", "SHAP 模型解释", "外部验证"]
            if st.session_state.get(key, False):
                st.success(f"✅ Step {i}: {labels[i-1]}")
            else:
                st.warning(f"⏳ Step {i}: {labels[i-1]}")
        st.markdown("---")
        palette = st.selectbox("🎨 选择 SCI 调色盘", list(PALETTES.keys()), index=0, key="global_palette")
        st.session_state.palette_choice = palette

        if st.session_state.all_outputs:
            st.markdown("---")
            st.subheader("📦 一键导出所有结果")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in st.session_state.all_outputs:
                    zf.writestr(fname, data)
            st.download_button("📥 下载全部结果 (ZIP)", zip_buf.getvalue(), "Full_Analysis_Results.zip", "application/zip", key="global_download")
            if st.button("🗑️ 清空所有结果"):
                st.session_state.all_outputs = []
                st.rerun()

# ==================== 主工作区 ====================
if mode == "主队列完整分析":
    # ---------- 第一步 ----------
    st.header("第一步：数据上传、预处理与基线表")
    uploaded = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"], key="main_upload")
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            df.columns = [clean_column_name(c) for c in df.columns]
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].apply(lambda x: clean_cell(x) if pd.notnull(x) else x)
            st.session_state.df_raw = df.copy()
            st.success("文件加载并清理完成。")
        except Exception as e:
            st.error(f"读取失败: {e}")
            st.stop()

        st.subheader("原始数据预览")
        st.dataframe(df.head())

        # 缺失值处理
        st.subheader("缺失值处理策略")
        missing_strategy = st.selectbox(
            "选择插补方法",
            ["None (keep NA)", "Drop rows with NA", "Mean/Median fill", "KNN Impute", "Multiple Imputation (MICE)"],
            index=2
        )
        cat_cols = st.multiselect("指定分类变量（用于编码和填充）", options=df.columns.tolist(), default=[])
        st.session_state.cat_cols = cat_cols

        if st.button("✔️ 应用预处理"):
            df_clean = st.session_state.df_raw.copy()
            from sklearn.preprocessing import LabelEncoder

            encoders = {}
            for col in cat_cols:
                if col in df_clean.columns:
                    le = LabelEncoder()
                    # 转为 str 以确保一致性，再 fit_transform
                    df_clean[col] = le.fit_transform(df[col].astype(str)).astype(int)  # 转为整数
                    encoders[col] = le
            st.session_state.encoders = encoders
            if missing_strategy == "Drop rows with NA":
                df_clean.dropna(inplace=True)
            elif missing_strategy == "Mean/Median fill":
                for col in df_clean.select_dtypes(include=np.number).columns:
                    if df_clean[col].isnull().any():
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                for col in cat_cols:
                    if col in df_clean.columns and df_clean[col].isnull().any():
                        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)
            elif missing_strategy == "KNN Impute":
                from sklearn.impute import KNNImputer
                num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    imputer = KNNImputer(n_neighbors=5)
                    df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
                for col in cat_cols:
                    if col in df_clean.columns and df_clean[col].isnull().any():
                        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)
            elif missing_strategy == "Multiple Imputation (MICE)":
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                from sklearn.linear_model import BayesianRidge
                num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
                    df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
                for col in cat_cols:
                    if col in df_clean.columns and df_clean[col].isnull().any():
                        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)

            st.session_state.df_clean = df_clean
            st.success("预处理完成！")
            st.dataframe(df_clean.head())

            # 缺失值报告
            na_cnt = df_clean.isnull().sum()
            na_df = pd.DataFrame({'Count': na_cnt, 'Percentage': (na_cnt/len(df_clean))*100})
            st.write("缺失值报告：", na_df[na_df['Count'] > 0] if na_df['Count'].sum() > 0 else "无缺失值")

        # 基线表（独立于预处理按钮，只要 df_clean 存在就显示）
        if st.session_state.df_clean is not None:
            st.markdown("---")
            st.subheader("📊 基线特征表 (Table 1)")
            all_vars = st.session_state.df_clean.columns.tolist()
            group_col = st.selectbox("选择分组变量（可选）", ["无"] + all_vars, key="baseline_group")
            selected_vars = st.multiselect("选择需要比较的变量（留空则全部纳入）", all_vars, default=all_vars, key="baseline_vars")
            if st.button("生成基线表", key="generate_baseline"):
                df_bl = st.session_state.df_clean
                cat_vars = st.session_state.cat_cols
                if group_col != "无":
                    groups = sorted(df_bl[group_col].dropna().unique())
                    table = []
                    for var in selected_vars:
                        if var == group_col:
                            continue
                        if var in cat_vars:
                            for level in sorted(df_bl[var].dropna().unique()):
                                row = [f"{var} = {level}"]
                                for g in groups:
                                    sub = df_bl[df_bl[group_col] == g]
                                    n = sub[var].eq(level).sum()
                                    rate = n/len(sub)*100 if len(sub)>0 else 0
                                    row.append(f"{n} ({rate:.1f}%)")
                                ct = pd.crosstab(df_bl[group_col], df_bl[var] == level)
                                if ct.shape == (2,2):
                                    _, p, _, _ = stats.chi2_contingency(ct)
                                    row.append(f"{p:.4f}")
                                else:
                                    row.append("N/A")
                                table.append(row)
                        else:
                            row = [var]
                            vals = [df_bl[df_bl[group_col] == g][var].dropna() for g in groups]
                            for v in vals:
                                row.append(f"{v.mean():.2f} ± {v.std():.2f}")
                            if len(groups) == 2:
                                _, p = stats.ttest_ind(*vals)
                            else:
                                _, p = stats.f_oneway(*vals)
                            row.append(f"{p:.4f}")
                            table.append(row)
                    col_names = ["Variable"] + [f"{g} (n={df_bl[group_col].eq(g).sum()})" for g in groups] + ["P-value"]
                    baseline_df = pd.DataFrame(table, columns=col_names)
                    st.session_state.baseline_df = baseline_df
                    store_dataframe(baseline_df, "Baseline_Table.csv")
                else:
                    st.info("未选择分组变量，仅展示描述统计。")
                    baseline_df = df_bl[selected_vars].describe().T.reset_index().rename(columns={"index":"Variable"})
                    st.session_state.baseline_df = baseline_df

            # 若 baseline_df 已生成，则持续显示
            if st.session_state.baseline_df is not None:
                st.dataframe(st.session_state.baseline_df)
                st.download_button("📥 下载基线表 CSV", st.session_state.baseline_df.to_csv(index=False), "baseline.csv")

            # 第一步完成标记
            if st.session_state.baseline_df is not None:
                st.session_state.step1_done = True
            else:
                st.info("请至少生成一次基线表以完成第一步。")

    # ---------- 第二步（回归分析） ----------
    if st.session_state.get('step1_done', False):
        st.markdown("---")
        st.header("第二步：Logistic / Cox 回归与动态累积迭代分析")
        try:
            from utils.regression import run_univariate_regression, run_multivariate_regression, plot_forest_chart
        except ImportError:
            st.error("缺少回归分析模块 (utils/regression.py)，请先创建该模块。")
            st.stop()

        df = st.session_state.df_clean
        all_cols = df.columns.tolist()

        st.markdown("### ⚙️ 全局回归模型参数配置")
        reg_mode = st.selectbox("选择底层回归模型", ["Logistic 回归", "Cox 比例风险生存回归"])
        time_var, y_var = None, None
        col1, col2 = st.columns(2)
        with col1:
            if reg_mode == "Logistic 回归":
                y_var = st.selectbox("二分类因变量 (Y)", all_cols)
            else:
                time_var = st.selectbox("生存时间变量 (Time)", all_cols)
        with col2:
            if reg_mode == "Cox 比例风险生存回归":
                y_var = st.selectbox("生存结局变量 (Status)", all_cols)

        exclude = [y_var, time_var] if time_var else [y_var]
        potential_x = [c for c in all_cols if c not in exclude]
        palette = st.session_state.palette_choice

        tab1, tab2, tab3 = st.tabs(["📊 单因素分析", "📈 多因素分析", "🔄 累积迭代模型"])
        with tab1:
            st.markdown("#### 单因素回归分析筛选")
            uni_all = st.checkbox("一键全选", value=False, key="uni_all")
            default_uni = potential_x if uni_all else []
            selected_uni = st.multiselect("选择变量", potential_x, default=default_uni, key="uni_x")
            if st.button("🚀 运行单因素分析并出图"):
                if not selected_uni:
                    st.warning("请至少选择一个自变量！")
                else:
                    mtype = "Logistic" if reg_mode == "Logistic 回归" else "Cox"
                    uni_res = run_univariate_regression(df, y_var, selected_uni, model_type=mtype, time_var=time_var)
                    st.subheader("单因素回归结果表")
                    st.dataframe(uni_res.style.format(precision=4))
                    metric_col = "Odds Ratio (OR)" if mtype == "Logistic" else "Hazard Ratio (HR)"
                    fig = plot_forest_chart(uni_res, metric_col, palette_name=palette)
                    st.pyplot(fig)
                    store_dataframe(uni_res, "Univariate_Regression_Results.csv")
                    store_figure(fig, "Univariate_Forest.pdf")
                    st.session_state.step2_done = True

        with tab2:
            st.markdown("#### 多因素回归模型构建")
            multi_all = st.checkbox("一键全选", value=False, key="multi_all")
            default_multi = potential_x if multi_all else []
            selected_multi = st.multiselect("选择共同进入多因素模型的变量", potential_x, default=default_multi, key="multi_x")
            if st.button("🚀 运行多因素分析并出图"):
                if not selected_multi:
                    st.warning("请选择自变量组合！")
                else:
                    mtype = "Logistic" if reg_mode == "Logistic 回归" else "Cox"
                    multi_res = run_multivariate_regression(df, y_var, selected_multi, model_type=mtype, time_var=time_var)
                    st.subheader("多因素强校正结果表")
                    st.dataframe(multi_res.style.format(precision=4))
                    metric_col = "Odds Ratio (OR)" if mtype == "Logistic" else "Hazard Ratio (HR)"
                    fig = plot_forest_chart(multi_res, metric_col, palette_name=palette)
                    st.pyplot(fig)
                    store_dataframe(multi_res, "Multivariate_Regression_Results.csv")
                    store_figure(fig, "Multivariate_Forest.pdf")
                    st.session_state.step2_done = True

        with tab3:
            st.markdown("#### 🔄 临床变量动态累积迭代模型")
            if 'iter_vars_pool' not in st.session_state:
                st.session_state.iter_vars_pool = []
            col_a, col_b = st.columns([3,1])
            with col_a:
                available = [c for c in potential_x if c not in st.session_state.iter_vars_pool]
                next_var = st.selectbox("选择要追加的变量", options=available if available else ["无可用变量"])
            with col_b:
                if st.button("➕ 确认追加") and available:
                    st.session_state.iter_vars_pool.append(next_var)
                    st.rerun()
            if st.session_state.iter_vars_pool:
                st.info(f"当前变量序列: {' ➡️ '.join(st.session_state.iter_vars_pool)}")
                if st.button("🗑️ 清空当前迭代模型"):
                    st.session_state.iter_vars_pool = []
                    st.rerun()
                mtype = "Logistic" if reg_mode == "Logistic 回归" else "Cox"
                iter_res = run_multivariate_regression(df, y_var, st.session_state.iter_vars_pool, model_type=mtype, time_var=time_var)
                st.subheader("当前迭代轮次模型表现")
                st.dataframe(iter_res.style.format(precision=4))
                metric_col = "Odds Ratio (OR)" if mtype == "Logistic" else "Hazard Ratio (HR)"
                fig = plot_forest_chart(iter_res, metric_col, palette_name=palette)
                st.pyplot(fig)
                store_dataframe(iter_res, f"Iteration_Model_{len(st.session_state.iter_vars_pool)}_Results.csv")
                store_figure(fig, f"Iteration_Model_{len(st.session_state.iter_vars_pool)}_Forest.pdf")
                st.session_state.step2_done = True

    # ---------- 第三步（特征筛选） ----------
    if st.session_state.get('step2_done', False):
        st.markdown("---")
        st.header("第三步：全因素 LASSO 与 Boruta 双重特征筛选")
        try:
            from utils.feature_selection import (
                run_lasso_selection, run_boruta_selection,
                plot_feature_comparison, plot_lasso_path,
                plot_boruta_importance_distribution,
                plot_venn_diagram, plot_correlation_heatmap
            )
        except ImportError:
            st.error("缺少特征筛选模块 (utils/feature_selection.py)，请先创建该模块。")
            st.stop()

        df = st.session_state.df_clean
        all_cols = df.columns.tolist()

        st.markdown("### 🧬 特征筛选数据矩阵构建")
        y_selector = st.selectbox("确认用于特征筛选的因变量", all_cols, index=0)
        X_cols = [c for c in all_cols if c != y_selector]
        st.session_state.y_selector = y_selector

        complete_df = df[[y_selector] + X_cols].dropna()
        st.session_state.complete_df = complete_df
        X = complete_df[X_cols]
        y = complete_df[y_selector]

        st.markdown("### ⚙️ Lasso 惩罚力度选择")
        alpha_strategy = st.radio(
            "选择 lambda 策略:",
            ["lambda.min", "lambda.1se"],
            index=0, horizontal=True,
            help="lambda.min：最优预测；lambda.1se：更稀疏的模型"
        )

        col_run1, col_run2 = st.columns(2)
        with col_run1:
            palette_s3 = st.selectbox("选择调色盘", list(PALETTES.keys()), key="s3_pal")
        with col_run2:
            start_filter = st.button("🚀 一键触发双算法交叉降维")

        st.markdown("### 📊 扩展可视化（可选）")
        show_path = st.checkbox("LASSO 系数路径图", value=True)
        show_dist = st.checkbox("Boruta 重要性分布图", value=True)
        show_venn = st.checkbox("Venn 图", value=True)
        show_heatmap = st.checkbox("特征相关性热图", value=True)

        if start_filter or 'lasso_feats' in st.session_state:
            with st.spinner("双引擎并发计算中..."):
                # 缺失值安全处理
                if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                    st.warning("检测到缺失值，已自动填充。")
                    for col in X.columns:
                        if X[col].dtype in ['float64', 'int64']:
                            X[col].fillna(X[col].median(), inplace=True)
                        else:
                            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown", inplace=True)
                    if y.isnull().sum() > 0:
                        y.fillna(y.mode()[0] if not y.mode().empty else 0, inplace=True)

                if 'lasso_feats' not in st.session_state:
                    lasso_feats, lasso_imp = run_lasso_selection(
                        X, y, cv=5, alpha_strategy=alpha_strategy, random_state=42
                    )
                    boruta_feats, tentative_feats, boruta_imp = run_boruta_selection(X, y)
                    st.session_state.update({
                        'lasso_feats': lasso_feats,
                        'lasso_imp': lasso_imp,
                        'boruta_feats': boruta_feats,
                        'tentative_feats': tentative_feats,
                        'boruta_imp': boruta_imp
                    })

                lf = st.session_state.lasso_feats
                bf = st.session_state.boruta_feats
                both = list(set(lf) & set(bf))
                union = list(set(lf) | set(bf))

                st.success("✨ 特征筛选完成！")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("LASSO 检出变量", f"{len(lf)} 个")
                col_m2.metric("Boruta 确认变量", f"{len(bf)} 个")
                col_m3.metric("双阳性交集", f"{len(both)} 个")

                st.markdown("### 🎯 入组变量策略")
                strategy = st.radio(
                    "选择最终特征集：",
                    [
                        f"仅 LASSO 阳性 ({len(lf)}个)",
                        f"仅 Boruta 阳性 ({len(bf)}个)",
                        f"双阳性交集 ({len(both)}个) [推荐]",
                        f"并集 ({len(union)}个)"
                    ]
                )
                if "仅 LASSO" in strategy:
                    st.session_state.final_ml_features = lf
                elif "仅 Boruta" in strategy:
                    st.session_state.final_ml_features = bf
                elif "交集" in strategy:
                    st.session_state.final_ml_features = both
                else:
                    st.session_state.final_ml_features = union

                st.info(f"📋 锁定特征: {', '.join(st.session_state.final_ml_features)}")

                pal = get_palette(palette_s3)
                fig_fs = plot_feature_comparison(st.session_state.lasso_imp, st.session_state.boruta_imp, strategy, both, pal)
                st.pyplot(fig_fs)
                store_figure(fig_fs, "Feature_Selection_Comparison.pdf")
                store_dataframe(st.session_state.lasso_imp, "LASSO_Importance.csv")
                store_dataframe(st.session_state.boruta_imp, "Boruta_Importance.csv")

                if show_path:
                    with st.spinner("绘制 LASSO 路径图..."):
                        fig_path = plot_lasso_path(X, y, alphas=50, palette_colors=pal)
                        st.pyplot(fig_path)
                        store_figure(fig_path, "LASSO_Path.pdf")
                if show_dist:
                    with st.spinner("绘制 Boruta 分布图..."):
                        fig_dist = plot_boruta_importance_distribution(st.session_state.boruta_imp, palette_colors=pal)
                        st.pyplot(fig_dist)
                        store_figure(fig_dist, "Boruta_Distribution.pdf")
                if show_venn:
                    with st.spinner("绘制 Venn 图..."):
                        fig_venn = plot_venn_diagram(lf, bf, None, palette_colors=pal)
                        st.pyplot(fig_venn)
                        store_figure(fig_venn, "Venn_Diagram.pdf")
                if show_heatmap:
                    with st.spinner("绘制相关性热图..."):
                        if len(st.session_state.final_ml_features) > 1:
                            fig_heat = plot_correlation_heatmap(X, st.session_state.final_ml_features, palette_colors=pal)
                            st.pyplot(fig_heat)
                            store_figure(fig_heat, "Correlation_Heatmap.pdf")
                        else:
                            st.warning("特征数量少于2，无法绘制热图。")
                st.session_state.step3_done = True
                st.success("✅ 特征筛选完成，已解锁下一步机器学习建模。")

    # 后续步骤（第四、五、六步）可依此类推
                # ================== 第四步：机器学习 ==================
                if st.session_state.step3_done:
                    st.markdown("---")
                    st.header("第四步：集成机器学习模型构建与全方位评价矩阵")

                    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
                    from utils.ml_models import (
                        get_model_dict, evaluate_predictions,
                        plot_multi_roc, plot_metric_bars, plot_confusion_matrices,
                        calculate_univariate_auc, plot_univariate_auc_bar,
                        plot_forest_style, plot_scatter_comparison, plot_radar_chart,
                        plot_univariate_auc_forest, plot_univariate_roc_curves, plot_calibration_curve, plot_decision_curve  # 新增
                    )

                    final_features = st.session_state.final_ml_features
                    y_col = st.session_state.y_selector  # 从第三步继承

                    if len(final_features) == 0:
                        st.error("警告：您在第三步没有选中任何特征！无法进行机器学习。请回退修改。")
                    else:
                        # 使用第三步保存的 complete_df
                        complete_df = st.session_state.complete_df
                        X = complete_df[final_features]
                        y = complete_df[y_col]

                        # ---------- 单变量 AUC 分析（增强图表类型选择） ----------
                        st.markdown("### 📊 单个特征独立 ROC 曲线分析")
                        with st.spinner("正在绘制每个特征的 ROC 曲线..."):
                            # 直接调用 ROC 曲线绘制函数，传入原始 X 和 y
                            fig_univar_roc = plot_univariate_roc_curves(X, y, feature_names=final_features,
                                                                        palette_name=st.session_state.palette_choice)
                            if fig_univar_roc is not None:
                                st.pyplot(fig_univar_roc)
                                store_figure(fig_univar_roc, "Univariate_ROC_Curves.pdf")
                            else:
                                st.warning("没有特征可用于绘制 ROC 曲线。")
                        # ---------- 数据拆分 ----------
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                                            stratify=y)
                        st.markdown(
                            f"**数据拆分完成：** 训练集 {len(X_train)} 例，验证集 {len(X_test)} 例。入组特征：{len(final_features)} 个。")

                        # ---------- 模型选择 ----------
                        all_models_dict = get_model_dict()
                        st.markdown("### 🤖 请选择要构建的监督学习模型")
                        col_sel1, col_sel2 = st.columns([4, 1])
                        with col_sel1:
                            selected_model_names = st.multiselect(
                                "支持多选，选中的模型将进行全方位 PK",
                                list(all_models_dict.keys()),
                                default=["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
                            )
                        with col_sel2:
                            run_ml_btn = st.button("🚀 一键训练与全矩阵分析", type="primary")

                        # ---------- 训练逻辑（仅在点击按钮时执行） ----------
                        if run_ml_btn and selected_model_names:
                            with st.spinner("算力引擎全开：正在进行模型拟合、重抽样 CI 计算与交叉验证..."):
                                train_metrics, test_metrics = [], []
                                train_probs, test_probs = {}, {}
                                train_preds, test_preds = {}, {}
                                cv_10_results = []

                                progress_bar = st.progress(0)
                                for idx, m_name in enumerate(selected_model_names):
                                    model = all_models_dict[m_name]
                                    model.fit(X_train, y_train)

                                    y_train_pred = model.predict(X_train)
                                    y_train_prob = model.predict_proba(X_train)[:, 1]
                                    train_probs[m_name] = y_train_prob
                                    train_preds[m_name] = y_train_pred
                                    train_metrics.append(
                                        evaluate_predictions(y_train, y_train_pred, y_train_prob, m_name))

                                    y_test_pred = model.predict(X_test)
                                    y_test_prob = model.predict_proba(X_test)[:, 1]
                                    test_probs[m_name] = y_test_prob
                                    test_preds[m_name] = y_test_pred
                                    test_metrics.append(evaluate_predictions(y_test, y_test_pred, y_test_prob, m_name))

                                    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                                    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                                    cv_10_results.append({
                                        "Model": m_name,
                                        "10_Fold_Mean_AUC": np.mean(cv_aucs),
                                        "10_Fold_Std": np.std(cv_aucs)
                                    })
                                    progress_bar.progress((idx + 1) / len(selected_model_names))

                                st.success("全部模型训练及审计完毕！所有结果图表均自动生成。")

                                # 构建 DataFrames
                                df_test_metrics = pd.DataFrame(test_metrics)
                                df_train_metrics = pd.DataFrame(train_metrics)
                                df_cv10 = pd.DataFrame(cv_10_results)

                                # 存入 session_state（用于后续展示，避免重训练）
                                st.session_state['df_test_metrics'] = df_test_metrics
                                st.session_state['df_train_metrics'] = df_train_metrics
                                st.session_state['df_cv10'] = df_cv10
                                st.session_state['train_probs'] = train_probs
                                st.session_state['test_probs'] = test_probs
                                st.session_state['train_preds'] = train_preds
                                st.session_state['test_preds'] = test_preds
                                st.session_state['y_train'] = y_train
                                st.session_state['y_test'] = y_test

                                store_dataframe(df_train_metrics, "Metrics_Train.csv")
                                store_dataframe(df_test_metrics, "Metrics_Validation.csv")
                                store_dataframe(df_cv10, "10_Fold_CV_Results.csv")

                                # 自动提取 AUC 最高的模型
                                best_model_name = df_test_metrics.loc[df_test_metrics['AUC'].idxmax()]['Model']
                                st.session_state.best_ml_model = all_models_dict[best_model_name]
                                st.session_state.best_ml_name = best_model_name
                                st.session_state.X_train_shap = X_train
                                st.session_state.step4_done = True

                                # 强制刷新以显示结果
                                st.rerun()

                        # ---------- 结果展示（仅当 session_state 中有训练结果时） ----------
                        if 'df_test_metrics' in st.session_state:
                            df_test_metrics = st.session_state.df_test_metrics
                            df_train_metrics = st.session_state.df_train_metrics
                            df_cv10 = st.session_state.df_cv10
                            train_probs = st.session_state.train_probs
                            test_probs = st.session_state.test_probs
                            train_preds = st.session_state.train_preds
                            test_preds = st.session_state.test_preds
                            y_train = st.session_state.y_train
                            y_test = st.session_state.y_test

                            # 在 Tabs 定义中添加（示例）：
                            tab_roc, tab_bar, tab_cm, tab_cal, tab_table, tab_cv = st.tabs([
                                "📈 综合 ROC 曲线",
                                "📊 模型性能对比",
                                "🔲 混淆矩阵热力图",
                                "📐 校准与决策曲线",  # 新增
                                "📑 全评价指标详细表",
                                "🔄 内部 10-Fold 验证"
                            ])

                            with tab_cal:
                                # ========== 校准曲线 (Calibration Curves) ==========
                                st.markdown("#### 校准曲线 (Calibration Curves)")
                                col_cal1, col_cal2 = st.columns(2)
                                best_model_name = df_test_metrics.loc[df_test_metrics['AUC'].idxmax()]['Model']

                                with col_cal1:
                                    st.markdown("**训练集 (Training Set)**")
                                    if best_model_name in train_probs:
                                        fig_cal_train = plot_calibration_curve(y_train, train_probs[best_model_name],
                                                                               best_model_name + " (Train)",
                                                                               palette_name=st.session_state.palette_choice)
                                        st.pyplot(fig_cal_train)
                                        store_figure(fig_cal_train, "Calibration_Curve_Train.pdf")
                                    else:
                                        st.warning("无训练集预测概率。")

                                with col_cal2:
                                    st.markdown("**验证集 (Validation Set)**")
                                    if best_model_name in test_probs:
                                        fig_cal_test = plot_calibration_curve(y_test, test_probs[best_model_name],
                                                                              best_model_name + " (Validation)",
                                                                              palette_name=st.session_state.palette_choice)
                                        st.pyplot(fig_cal_test)
                                        store_figure(fig_cal_test, "Calibration_Curve_Test.pdf")
                                    else:
                                        st.warning("无验证集预测概率。")

                                # ========== 决策曲线 (Decision Curve Analysis) ==========
                                st.markdown("#### 决策曲线 (Decision Curve Analysis)")
                                col_dca1, col_dca2 = st.columns(2)

                                with col_dca1:
                                    st.markdown("**训练集 (Training Set)**")
                                    if best_model_name in train_probs:
                                        fig_dca_train = plot_decision_curve(y_train, train_probs[best_model_name],
                                                                            best_model_name + " (Train)",
                                                                            palette_name=st.session_state.palette_choice)
                                        st.pyplot(fig_dca_train)
                                        store_figure(fig_dca_train, "Decision_Curve_Train.pdf")
                                    else:
                                        st.warning("无训练集预测概率。")

                                with col_dca2:
                                    st.markdown("**验证集 (Validation Set)**")
                                    if best_model_name in test_probs:
                                        fig_dca_test = plot_decision_curve(y_test, test_probs[best_model_name],
                                                                           best_model_name + " (Validation)",
                                                                           palette_name=st.session_state.palette_choice)
                                        st.pyplot(fig_dca_test)
                                        store_figure(fig_dca_test, "Decision_Curve_Test.pdf")
                                    else:
                                        st.warning("无验证集预测概率。")

                            # ---------- Tab: ROC ----------
                            with tab_roc:
                                col_r1, col_r2 = st.columns(2)
                                with col_r1:
                                    fig_roc_train = plot_multi_roc(y_train, train_probs, "Training", st.session_state.palette_choice)
                                    st.pyplot(fig_roc_train)
                                    store_figure(fig_roc_train, "ROC_Train.pdf")
                                with col_r2:
                                    fig_roc_test = plot_multi_roc(y_test, test_probs, "Validation", st.session_state.palette_choice)
                                    st.pyplot(fig_roc_test)
                                    store_figure(fig_roc_test, "ROC_Validation.pdf")

                            # ---------- Tab: 模型性能对比（多种图表） ----------
                            with tab_bar:
                                st.markdown("#### 模型性能对比")

                                # 选择数据集：训练集 或 验证集
                                dataset_choice = st.radio(
                                    "选择数据集",
                                    options=["训练集 (Training)", "验证集 (Validation)"],
                                    horizontal=True
                                )

                                # 根据选择获取对应的指标 DataFrame
                                if dataset_choice == "训练集 (Training)":
                                    metrics_df = df_train_metrics
                                    dataset_label = "Training"
                                else:
                                    metrics_df = df_test_metrics
                                    dataset_label = "Validation"

                                # 选择要对比的指标
                                metric_target = st.selectbox(
                                    "选择要对比的指标",
                                    ["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"],
                                    key="metric_target"
                                )

                                # 选择图表类型
                                chart_type = st.selectbox(
                                    "选择图表类型",
                                    ["柱状图 (Bar)", "森林图 (Forest)", "散点图 (Scatter)", "雷达图 (Radar)"],
                                    key="chart_type"
                                )

                                fig = None
                                if chart_type == "柱状图 (Bar)":
                                    fig = plot_metric_bars(metrics_df, metric_target, dataset_label, st.session_state.palette_choice)
                                elif chart_type == "森林图 (Forest)":
                                    fig = plot_forest_style(metrics_df, metric_target, dataset_label, st.session_state.palette_choice)
                                elif chart_type == "散点图 (Scatter)":
                                    x_metric = st.selectbox(
                                        "X轴指标",
                                        ["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"],
                                        key="x_metric"
                                    )
                                    y_metric = st.selectbox(
                                        "Y轴指标",
                                        ["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"],
                                        key="y_metric",
                                        index=1
                                    )
                                    fig = plot_scatter_comparison(metrics_df, x_metric, y_metric, dataset_label,
                                                                  st.session_state.palette_choice)
                                elif chart_type == "雷达图 (Radar)":
                                    radar_metrics = st.multiselect(
                                        "选择雷达图指标",
                                        ["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"],
                                        default=["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"]
                                    )
                                    if len(radar_metrics) < 3:
                                        st.warning("请至少选择3个指标")
                                    else:
                                        fig = plot_radar_chart(metrics_df, radar_metrics, dataset_label, st.session_state.palette_choice)

                                if fig is not None:
                                    st.pyplot(fig)
                                    store_figure(fig, f"{chart_type}_{metric_target}_{dataset_label}.pdf")

                            # ---------- Tab: 混淆矩阵 ----------
                            with tab_cm:
                                st.markdown("#### 混淆矩阵 (Confusion Matrices)")
                                col_cm1, col_cm2 = st.columns(2)
                                with col_cm1:
                                    st.markdown("**训练集 (Training Set)**")
                                    fig_cm_train = plot_confusion_matrices(
                                        y_train, train_preds,
                                        title_prefix="Train"
                                    )
                                    st.pyplot(fig_cm_train)
                                    store_figure(fig_cm_train, "Confusion_Matrices_Train.pdf")
                                with col_cm2:
                                    st.markdown("**验证集 (Validation Set)**")
                                    fig_cm_test = plot_confusion_matrices(
                                        y_test, test_preds,
                                        title_prefix="Validation"
                                    )
                                    st.pyplot(fig_cm_test)
                                    store_figure(fig_cm_test, "Confusion_Matrices_Validation.pdf")
                                    
                            # ---------- Tab: 详细表格 ----------
                            with tab_table:
                                st.markdown("#### 训练集完整评价矩阵")
                                st.dataframe(df_train_metrics.style.format(precision=3))
                                st.markdown("#### 验证集完整评价矩阵（包含置信区间、最佳决断值 Youden Cutoff）")
                                st.dataframe(df_test_metrics.style.format(precision=3))

                            # ---------- Tab: 交叉验证 ----------
                            with tab_cv:
                                st.markdown("#### 全集内部十折交叉验证 (10-Fold Cross Validation)")
                                st.dataframe(df_cv10.style.format(precision=4))

                        # ---------- 第五步：SHAP 解释（仅在模型训练完成时显示） ----------
                        if 'best_ml_model' in st.session_state:
                            st.markdown("---")
                            st.header("第五步：最强王者模型黑盒揭秘 (SHAP 多维解释)")

                            from utils.shap_analyzer import (
                                get_explainer_and_values, plot_shap_bar, plot_shap_beeswarm,
                                plot_shap_heatmap, plot_shap_waterfall, plot_shap_force,
                                plot_shap_dependence, plot_interaction_matrix, plot_interaction_network
                            )

                            best_model = st.session_state.best_ml_model
                            best_name = st.session_state.best_ml_name
                            X_train_base = st.session_state.X_train_shap

                            st.success(f"🏆 自动捕获当前性能最高模型：**{best_name}**。正在启动全息 SHAP 解析...")

                            if 'shap_calculated' not in st.session_state:
                                with st.spinner("算力燃烧中... 正在进行博弈论 Shapley Value 拆解 (可能需要1-3分钟)..."):
                                    try:
                                        exp, vals, inter_vals, X_scaled = get_explainer_and_values(best_model,
                                                                                                   X_train_base,
                                                                                                   best_name)
                                        st.session_state.update({
                                            'shap_explainer': exp,
                                            'shap_values': vals,
                                            'shap_interactions': inter_vals,
                                            'X_scaled': X_scaled,
                                            'shap_calculated': True
                                        })
                                    except Exception as e:
                                        st.error(f"SHAP 解析遇到瓶颈：{str(e)}")
                                        st.stop()

                            if st.session_state.get('shap_calculated', False):
                                vals = st.session_state.shap_values
                                inter_vals = st.session_state.shap_interactions
                                feat_names = st.session_state.X_scaled.columns.tolist()

                                tab_global, tab_local, tab_dep, tab_interact = st.tabs([
                                    "🌍 全局重要性", "🔬 个体诊断溯源", "📉 依赖与热力分布", "🕸️ 联合交互作用"
                                ])

                                with tab_global:
                                    col_g1, col_g2 = st.columns(2)
                                    with col_g1:
                                        st.markdown("##### 全局特征重要性条形图")
                                        fig_bar = plot_shap_bar(vals)
                                        st.pyplot(fig_bar)
                                        store_figure(fig_bar, "SHAP_Global_Bar.pdf")
                                    with col_g2:
                                        st.markdown("##### 蜂群图")
                                        fig_bee = plot_shap_beeswarm(vals)
                                        st.pyplot(fig_bee)
                                        store_figure(fig_bee, "SHAP_Beeswarm.pdf")

                                with tab_local:
                                    st.markdown("##### 单个样本病理拆解")
                                    patient_idx = st.slider("请选择患者编号", 0, len(vals) - 1, 0)
                                    col_l1, col_l2 = st.columns(2)
                                    with col_l1:
                                        fig_wf = plot_shap_waterfall(vals, patient_idx)
                                        st.pyplot(fig_wf)
                                        store_figure(fig_wf, f"SHAP_Waterfall_Patient_{patient_idx}.pdf")
                                    with col_l2:
                                        fig_force = plot_shap_force(st.session_state.shap_explainer, vals, patient_idx)
                                        st.pyplot(fig_force)
                                        store_figure(fig_force, f"SHAP_Force_Patient_{patient_idx}.pdf")

                                with tab_dep:
                                    dep_feat = st.selectbox("选择特征", feat_names)
                                    fig_dep = plot_shap_dependence(vals, dep_feat)
                                    st.pyplot(fig_dep)
                                    store_figure(fig_dep, f"SHAP_Dependence_{dep_feat}.pdf")
                                    fig_hm = plot_shap_heatmap(vals)
                                    st.pyplot(fig_hm)
                                    store_figure(fig_hm, "SHAP_Heatmap.pdf")

                                with tab_interact:
                                    if inter_vals is not None:
                                        st.success("检测到树模型，已计算交互作用矩阵。")
                                        col_i1, col_i2 = st.columns(2)
                                        with col_i1:
                                            fig_imat = plot_interaction_matrix(inter_vals, feat_names)
                                            st.pyplot(fig_imat)
                                            store_figure(fig_imat, "SHAP_Interaction_Matrix.pdf")
                                        with col_i2:
                                            thresh = st.slider("网络图阈值", 0.0, 0.1, 0.01, step=0.005)
                                            fig_inet = plot_interaction_network(inter_vals, feat_names, thresh)
                                            st.pyplot(fig_inet)
                                            store_figure(fig_inet, "SHAP_Interaction_Network.pdf")
                                    else:
                                        st.warning("当前模型非树模型，无法计算交互作用。")

                                st.session_state.step5_done = True
                                st.info("💡 所有 SHAP 图表已存入全局结果池，请在左侧边栏点击下载全部结果。")
                        else:
                            st.info("💡 完成第四步的模型训练后，此处将自动解锁SHAP分析。")
                st.markdown("---")
                st.header("第六步：外部独立验证（External Validation）")

                if 'best_ml_model' not in st.session_state:
                    st.warning("请先完成第四步的模型训练！")
                else:
                    ext_file = st.file_uploader("请上传外部验证数据集（CSV或Excel）", type=['csv', 'xlsx'],
                                                key="ext_file")
                    if ext_file is not None:
                        # 读取数据
                        if ext_file.name.endswith('.csv'):
                            df_ext = pd.read_csv(ext_file)
                        else:
                            df_ext = pd.read_excel(ext_file)
                        st.success("外部数据上传成功！")

                        # 对齐特征（使用训练时的 final_features 和 y_col）
                        final_features = st.session_state.final_ml_features
                        y_col = st.session_state.y_selector

                        # 检查列是否存在
                        missing_cols = [c for c in final_features + [y_col] if c not in df_ext.columns]
                        if missing_cols:
                            st.error(f"外部数据缺少以下列：{missing_cols}")
                        else:
                            X_ext = df_ext[final_features]
                            y_ext = df_ext[y_col]
                            # 删除缺失值
                            ext_clean = df_ext[[y_col] + final_features].dropna()
                            X_ext_clean = ext_clean[final_features]
                            y_ext_clean = ext_clean[y_col]

                            # 预测
                            model = st.session_state.best_ml_model
                            y_ext_pred = model.predict(X_ext_clean)
                            y_ext_prob = model.predict_proba(X_ext_clean)[:, 1]

                            # 计算指标
                            ext_metrics = evaluate_predictions(y_ext_clean, y_ext_pred, y_ext_prob, "External")
                            st.dataframe(pd.DataFrame([ext_metrics]).style.format(precision=3))

                            # 绘制ROC
                            ext_probs_dict = {"External Model": y_ext_prob}
                            fig_roc_ext = plot_multi_roc(y_ext_clean, ext_probs_dict, "External Validation",
                                                         st.session_state.palette_choice)
                            st.pyplot(fig_roc_ext)
                            store_figure(fig_roc_ext, "External_ROC.pdf")

                            # 校准曲线
                            fig_cal_ext = plot_calibration_curve(y_ext_clean, y_ext_prob, "External", st.session_state.palette_choice)
                            st.pyplot(fig_cal_ext)
                            store_figure(fig_cal_ext, "External_Calibration.pdf")

                            # 决策曲线
                            fig_dca_ext = plot_decision_curve(y_ext_clean, y_ext_prob, "External", st.session_state.palette_choice)
                            st.pyplot(fig_dca_ext)
                            store_figure(fig_dca_ext, "External_DecisionCurve.pdf")

                            # 混淆矩阵
                            fig_cm_ext = plot_confusion_matrices(y_ext_clean, {"External": y_ext_pred})
                            st.pyplot(fig_cm_ext)
                            store_figure(fig_cm_ext, "External_ConfusionMatrix.pdf")

                            st.success("外部验证完成！所有图表已存入全局下载池。")


