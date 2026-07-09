import streamlit as st

def check_password():
    """返回 True 表示用户输入了正确的密码。"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    # 显示密码输入框
    password = st.text_input("请输入访问密码", type="password")
    if st.button("登录"):
        # 从 secrets 中读取正确密码进行比较
        if password == st.secrets["ACCESS_PASSWORD"]:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("😕 密码错误，请重试。")
    return False

# 如果密码验证失败，则停止执行后续代码
if not check_password():
    st.stop()

# ---------- 以下是您原有的应用代码 ----------
# st.title("🔬 临床数据智能分析与机器学习平台")
# ... 您的所有原有代码 ...密码我想设置为123456789
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import zipfile
from utils.data_cleaner import clean_column_names
from utils.baseline import generate_tableone
from utils.plot_style import get_palette, PALETTES

# 页面配置
st.set_page_config(page_title="临床多组学自动化分析平台", layout="wide")

# ========== 全局状态初始化 ==========
if 'data_uploaded' not in st.session_state:
    st.session_state.update({
        'data_uploaded': False,
        'step1_done': False,
        'step2_done': False,
        'step3_done': False,
        'step4_done': False,
        'step5_done': False,
        'df_clean': None,
        'all_outputs': [],
        'final_ml_features': [],
        'complete_df': None,  # 用于第四步
        'y_selector': None,
        'palette_choice': 'Nature'  # 全局调色盘
    })


# ========== 辅助存储函数 ==========
def store_figure(fig, filename):
    """
    存储 matplotlib 图形到全局 all_outputs。
    如果同名文件已存在，则替换（删除旧条目再添加）。
    同时保存 PDF 和 TIFF (300 DPI) 两个版本。
    """
    # 删除所有已存在的同名文件（不管扩展名）
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    st.session_state.all_outputs = [
        (f, d) for f, d in st.session_state.all_outputs
        if not f.startswith(base_name)  # 删除以 base_name 开头的所有条目
    ]

    # 保存 PDF
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
    st.session_state.all_outputs.append((f"{base_name}.pdf", buf_pdf.getvalue()))

    # 保存 TIFF (300 DPI)
    buf_tiff = io.BytesIO()
    fig.savefig(buf_tiff, format='tiff', dpi=300, bbox_inches='tight')
    st.session_state.all_outputs.append((f"{base_name}.tiff", buf_tiff.getvalue()))


def store_dataframe(df, filename):
    """存储 DataFrame 到全局 all_outputs，同名文件自动覆盖（只保留 CSV）"""
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    # 删除旧版本
    st.session_state.all_outputs = [
        (f, d) for f, d in st.session_state.all_outputs
        if not f.startswith(base_name)
    ]
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.session_state.all_outputs.append((f"{base_name}.csv", csv_bytes))


def store_string(text, filename):
    """存储文本，同名文件自动覆盖"""
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    st.session_state.all_outputs = [
        (f, d) for f, d in st.session_state.all_outputs
        if not f.startswith(base_name)
    ]
    st.session_state.all_outputs.append((f"{base_name}.txt", text.encode('utf-8')))


# ========== 国风 UI ==========
st.markdown("""
    <style>
    .stApp { background-color: #fdfbf7; }
    h1, h2, h3 { color: #4a4266; font-family: "Microsoft YaHei", sans-serif; }
    .stButton>button { background-color: #ff461f; color: white; border-radius: 5px; border: none; }
    .stButton>button:hover { background-color: #c3270b; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 临床数据智能分析与机器学习平台")
st.markdown(
    "一键完成：**基线表 -> 回归迭代 -> 筛选特征 (LASSO/Boruta) -> 机器学习 -> SHAP解释**。支持 CNS 级别可视化下载。")

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 分析流程导航")
    mode = st.radio("选择模式", ["主队列完整分析"])

    if mode == "主队列完整分析":
        st.markdown("---")
        # 步骤状态
        for i in range(1, 6):
            key = f'step{i}_done'
            label = ["数据预处理与基线表", "单/多因素回归与迭代", "LASSO/Boruta 特征筛选",
                     "机器学习建模与评估", "SHAP 模型解释"][i - 1]
            if st.session_state.get(key, False):
                st.success(f"✅ Step {i}: {label}")
            else:
                st.warning(f"⏳ Step {i}: {label}")

        st.markdown("---")
        # 统一调色盘选择
        palette_choice = st.selectbox(
            "🎨 选择 SCI 调色盘",
            list(PALETTES.keys()),
            index=0,
            key="global_palette"
        )
        st.session_state.palette_choice = palette_choice

        # 统一下载按钮
        if st.session_state.all_outputs:
            st.markdown("---")
            st.subheader("📦 一键导出所有结果")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in st.session_state.all_outputs:
                    zf.writestr(fname, data)
            st.download_button(
                label="📥 下载全部结果 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="Full_Analysis_Results.zip",
                mime="application/zip",
                key="global_download"
            )
            if st.button("🗑️ 清空所有结果"):
                st.session_state.all_outputs = []
                st.rerun()

# ========== 主工作区 ==========
if mode == "主队列完整分析":
    # ---------- 第一步 ----------
    st.header("第一步：数据上传与基线特征分析 (Table 1)")
    uploaded_file = st.file_uploader("请上传临床数据集 (CSV 或 Excel 格式)", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("数据上传成功！正在进行列名规范化处理...")
        df = clean_column_names(df)
        st.session_state.df_clean = df
        st.session_state.data_uploaded = True
        with st.expander("预览规范化后的数据", expanded=False):
            st.dataframe(df.head())

    if st.session_state.data_uploaded:
        st.markdown("### 📊 生成基线表 (Table 1)")
        df = st.session_state.df_clean
        all_cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            groupby_var = st.selectbox("请选择分组变量 (如：Outcome, Treatment)", options=["无分组"] + all_cols)
        with col2:
            categorical_vars = st.multiselect("请指定分类变量 (二分类/多分类，未选则默认视为连续变量)", options=all_cols)
        nonnormal_vars = st.multiselect("请指定非正态分布的连续变量 (将使用中位数和IQR)",
                                        options=[c for c in all_cols if c not in categorical_vars])

        if st.button("🚀 开始计算并生成基线表"):
            with st.spinner('正在自动应用统计学检验...'):
                mytable = generate_tableone(df, groupby_var, categorical_vars, nonnormal_vars)
                st.session_state.step1_done = True
                st.markdown("#### 基线分析结果")
                st.text(mytable.tabulate(headers="keys", tablefmt="github"))
                # 存储
                csv_bytes = mytable.to_csv().encode('utf-8')
                st.session_state.all_outputs.append(("Table1_Baseline.csv", csv_bytes))
                txt = mytable.tabulate(headers="keys", tablefmt="github")
                store_string(txt, "Table1_Baseline.txt")
                # 独立下载
                st.download_button(
                    label="📥 下载基线表 (CSV)",
                    data=csv_bytes,
                    file_name="Table1_Baseline.csv",
                    mime="text/csv"
                )

    # ---------- 第二步 ----------
    if st.session_state.step1_done:
        st.markdown("---")
        st.header("第二步：Logistic / Cox 回归与动态累积迭代分析")
        from utils.regression import run_univariate_regression, run_multivariate_regression, plot_forest_chart

        df = st.session_state.df_clean
        all_cols = df.columns.tolist()

        st.markdown("### ⚙️ 全局回归模型参数配置")
        reg_mode = st.selectbox("请选择分析所用的底层回归模型", ["Logistic 回归", "Cox 比例风险生存回归"])
        time_var, y_var = None, None
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            if reg_mode == "Logistic 回归":
                y_var = st.selectbox("请选择二分类因变量 (Y, 如 Outcome)", options=all_cols)
            else:
                time_var = st.selectbox("请选择生存时间变量 (Time, 如 Survival_Months)", options=all_cols)
        with col_y2:
            if reg_mode == "Cox 比例风险生存回归":
                y_var = st.selectbox("请选择生存结局变量 (Status, 如 Event_Status)", options=all_cols)

        # 排除因变量
        exclude = [y_var, time_var] if time_var else [y_var]
        potential_x = [c for c in all_cols if c not in exclude]

        # 调色盘（使用全局）
        palette_choice = st.session_state.palette_choice

        tab_uni, tab_multi, tab_iter = st.tabs(
            ["📊 单因素分析", "📈 多因素分析", "🔄 累积迭代模型"]
        )

        # ---- 单因素 ----
        with tab_uni:
            st.markdown("#### 单因素回归分析筛选")
            uni_all = st.checkbox("一键全选", value=False, key="uni_all")
            default_uni = potential_x if uni_all else []
            selected_uni = st.multiselect("选择变量", potential_x, default=default_uni, key="uni_x")
            if st.button("🚀 运行单因素分析并出图"):
                if not selected_uni:
                    st.warning("请至少选择一个自变量！")
                else:
                    m_type = "Logistic" if reg_mode == "Logistic 回归" else "Cox"
                    uni_res = run_univariate_regression(df, y_var, selected_uni, model_type=m_type, time_var=time_var)
                    st.subheader("单因素回归结果表")
                    st.dataframe(uni_res.style.format(precision=4))
                    metric_col = "Odds Ratio (OR)" if m_type == "Logistic" else "Hazard Ratio (HR)"
                    fig = plot_forest_chart(uni_res, metric_col, palette_name=palette_choice)
                    st.pyplot(fig)
                    store_dataframe(uni_res, "Univariate_Regression_Results.csv")
                    store_figure(fig, "Univariate_Forest.pdf")
                    st.session_state.step2_done = True  # 解锁第三步

        # ---- 多因素 ----
        with tab_multi:
            st.markdown("#### 多因素回归模型构建")
            multi_all = st.checkbox("一键全选", value=False, key="multi_all")
            default_multi = potential_x if multi_all else []
            selected_multi = st.multiselect("选择共同进入多因素模型的变量", potential_x, default=default_multi,
                                            key="multi_x")
            if st.button("🚀 运行多因素分析并出图"):
                if not selected_multi:
                    st.warning("请选择自变量组合！")
                else:
                    m_type = "Logistic" if reg_mode == "Logistic 回归" else "Cox"
                    multi_res = run_multivariate_regression(df, y_var, selected_multi, model_type=m_type,
                                                            time_var=time_var)
                    st.subheader("多因素强校正结果表")
                    st.dataframe(multi_res.style.format(precision=4))
                    metric_col = "Odds Ratio (OR)" if m_type == "Logistic" else "Hazard Ratio (HR)"
                    fig = plot_forest_chart(multi_res, metric_col, palette_name=palette_choice)
                    st.pyplot(fig)
                    store_dataframe(multi_res, "Multivariate_Regression_Results.csv")
                    store_figure(fig, "Multivariate_Forest.pdf")
                    st.session_state.step2_done = True  # 解锁第三步

        # ---- 迭代 ----
        with tab_iter:
            st.markdown("#### 🔄 临床变量动态累积迭代模型")
            if 'iter_vars_pool' not in st.session_state:
                st.session_state.iter_vars_pool = []
            col_it1, col_it2 = st.columns([3, 1])
            with col_it1:
                available = [c for c in potential_x if c not in st.session_state.iter_vars_pool]
                next_var = st.selectbox("选择要追加的变量", options=available if available else ["无可用变量"])
            with col_it2:
                if st.button("➕ 确认追加") and available:
                    st.session_state.iter_vars_pool.append(next_var)
                    st.rerun()

            if st.session_state.iter_vars_pool:
                st.info(f"当前变量序列: {' ➡️ '.join(st.session_state.iter_vars_pool)}")
                if st.button("🗑️ 清空当前迭代模型"):
                    st.session_state.iter_vars_pool = []
                    st.rerun()
                m_type = "Logistic" if reg_mode == "Logistic 回归" else "Cox"
                iter_res = run_multivariate_regression(df, y_var, st.session_state.iter_vars_pool,
                                                       model_type=m_type, time_var=time_var)
                st.subheader("当前迭代轮次模型表现")
                st.dataframe(iter_res.style.format(precision=4))
                metric_col = "Odds Ratio (OR)" if m_type == "Logistic" else "Hazard Ratio (HR)"
                fig_iter = plot_forest_chart(iter_res, metric_col, palette_name=palette_choice)
                st.pyplot(fig_iter)
                iter_count = len(st.session_state.iter_vars_pool)
                store_dataframe(iter_res, f"Iteration_Model_{iter_count}_Results.csv")
                store_figure(fig_iter, f"Iteration_Model_{iter_count}_Forest.pdf")
                st.session_state.step2_done = True  # 解锁第三步
    # ---------- 第三步 ----------
    if st.session_state.step2_done:
        st.markdown("---")
        st.header("第三步：全因素 LASSO 与 Boruta 双重特征筛选")

        from utils.feature_selection import (
            run_lasso_selection, run_boruta_selection,
            plot_feature_comparison, plot_lasso_path,
            plot_boruta_importance_distribution,
            plot_venn_diagram, plot_correlation_heatmap
        )

        df = st.session_state.df_clean
        all_cols = df.columns.tolist()

        st.markdown("### 🧬 特征筛选数据矩阵构建")
        y_selector = st.selectbox("请确认用于特征筛选的因变量", options=all_cols, index=0)
        X_cols = [c for c in all_cols if c != y_selector]
        st.session_state.y_selector = y_selector

        # ===== 在这里定义 X 和 y（与 df 同级） =====
        complete_df = df[[y_selector] + X_cols].dropna()
        st.session_state.complete_df = complete_df
        X = complete_df[X_cols]
        y = complete_df[y_selector]

        # 调色板选择
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            palette_choice_s3 = st.selectbox("选择调色盘", list(PALETTES.keys()), key="s3_pal")
        with col_run2:
            start_filter = st.button("🚀 一键触发双算法交叉降维")

        # 扩展可视化复选框（建议放在按钮外，但放在这里也行）
        st.markdown("---")
        st.markdown("### 📊 扩展可视化图表（出版级）")
        show_lasso_path = st.checkbox("显示 LASSO 系数路径图", value=True)
        show_boruta_dist = st.checkbox("显示 Boruta 重要性分布图", value=True)
        show_venn = st.checkbox("显示 Venn 图", value=True)
        show_heatmap = st.checkbox("显示特征相关性热图", value=True)

        if start_filter or 'lasso_feats' in st.session_state:
            with st.spinner("双引擎并发计算中..."):
                # 现在 X 和 y 已定义，可以安全调用
                if 'lasso_feats' not in st.session_state:
                    lasso_feats, lasso_imp = run_lasso_selection(X, y)  # 正常工作
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
                both_positive = list(set(lf) & set(bf))
                union_positive = list(set(lf) | set(bf))

                st.success("✨ 特征筛选完成！")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("LASSO 检出变量", f"{len(lf)} 个")
                col_m2.metric("Boruta 确认变量", f"{len(bf)} 个")
                col_m3.metric("双阳性重叠变量 (交集)", f"{len(both_positive)} 个")

                st.markdown("### 🎯 机器学习入组变量策略抉择")
                strategy = st.radio(
                    "请选择用于下一步模型构建的最终特征集：",
                    [
                        f"仅使用 LASSO 阳性变量 ({len(lf)}个)",
                        f"仅使用 Boruta 确认阳性变量 ({len(bf)}个)",
                        f"双阳性严格交集变量 ({len(both_positive)}个) [🔥 推荐：最稳健]",
                        f"全集并集变量 ({len(union_positive)}个)"
                    ]
                )

                if "仅使用 LASSO" in strategy:
                    st.session_state.final_ml_features = lf
                elif "仅使用 Boruta" in strategy:
                    st.session_state.final_ml_features = bf
                elif "双阳性严格交集" in strategy:
                    st.session_state.final_ml_features = both_positive
                else:
                    st.session_state.final_ml_features = union_positive

                st.info(f"📋 当前锁定的入组特征序列: {', '.join(st.session_state.final_ml_features)}")

                # 主对比图
                pal_colors = get_palette(palette_choice_s3)
                fig_fs = plot_feature_comparison(
                    st.session_state.lasso_imp,
                    st.session_state.boruta_imp,
                    strategy, both_positive, pal_colors
                )
                st.pyplot(fig_fs)
                store_figure(fig_fs, "Feature_Selection_Comparison.pdf")
                store_dataframe(st.session_state.lasso_imp, "LASSO_Importance.csv")
                store_dataframe(st.session_state.boruta_imp, "Boruta_Importance.csv")

                # 扩展图
                if show_lasso_path:
                    with st.spinner("绘制 LASSO 路径图..."):
                        fig_path = plot_lasso_path(X, y, alphas=50, palette_colors=pal_colors)
                        st.pyplot(fig_path)
                        store_figure(fig_path, "LASSO_Path.pdf")
                if show_boruta_dist:
                    with st.spinner("绘制 Boruta 分布图..."):
                        fig_dist = plot_boruta_importance_distribution(
                            st.session_state.boruta_imp, palette_colors=pal_colors
                        )
                        st.pyplot(fig_dist)
                        store_figure(fig_dist, "Boruta_Distribution.pdf")
                if show_venn:
                    with st.spinner("绘制 Venn 图..."):
                        fig_venn = plot_venn_diagram(
                            st.session_state.lasso_feats,
                            st.session_state.boruta_feats,
                            st.session_state.tentative_feats,
                            palette_colors=pal_colors
                        )
                        st.pyplot(fig_venn)
                        store_figure(fig_venn, "Venn_Diagram.pdf")
                if show_heatmap:
                    with st.spinner("绘制相关性热图..."):
                        if len(st.session_state.final_ml_features) > 1:
                            fig_heat = plot_correlation_heatmap(
                                X, st.session_state.final_ml_features, palette_colors=pal_colors
                            )
                            st.pyplot(fig_heat)
                            store_figure(fig_heat, "Correlation_Heatmap.pdf")
                        else:
                            st.warning("选定的特征数量少于2，无法绘制热图。")

                st.session_state.step3_done = True
                st.success("✅ 特征筛选完成，已解锁下一步机器学习建模。")
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
                        # ---------- 单变量 AUC 分析（改为 ROC 曲线） ----------
                        st.markdown("### 📊 单个特征独立 ROC 曲线分析")
                        with st.spinner("正在绘制每个特征的 ROC 曲线..."):
                            # 直接调用 ROC 曲线绘制函数，传入原始 X 和 y
                            fig_univar_roc = plot_univariate_roc_curves(X, y, feature_names=final_features,
                                                                        palette_name=palette_choice)
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
                                st.markdown("#### 校准曲线")
                                # 训练集校准
                                best_model_name = df_test_metrics.loc[df_test_metrics['AUC'].idxmax()]['Model']
                                # 获取最佳模型的预测概率（需要从 train_probs 中提取）
                                if best_model_name in train_probs:
                                    fig_cal_train = plot_calibration_curve(y_train, train_probs[best_model_name],
                                                                           best_model_name + " (Train)",
                                                                           palette_name=palette_choice)
                                    st.pyplot(fig_cal_train)
                                    store_figure(fig_cal_train, "Calibration_Curve_Train.pdf")
                                # 验证集校准
                                if best_model_name in test_probs:
                                    fig_cal_test = plot_calibration_curve(y_test, test_probs[best_model_name],
                                                                          best_model_name + " (Validation)",
                                                                          palette_name=palette_choice)
                                    st.pyplot(fig_cal_test)
                                    store_figure(fig_cal_test, "Calibration_Curve_Test.pdf")

                                st.markdown("#### 决策曲线")
                                if best_model_name in test_probs:
                                    fig_dca = plot_decision_curve(y_test, test_probs[best_model_name],
                                                                  best_model_name + " (Validation)",
                                                                  palette_name=palette_choice)
                                    st.pyplot(fig_dca)
                                    store_figure(fig_dca, "Decision_Curve_Test.pdf")

                            # ---------- Tab: ROC ----------
                            with tab_roc:
                                col_r1, col_r2 = st.columns(2)
                                with col_r1:
                                    fig_roc_train = plot_multi_roc(y_train, train_probs, "Training", palette_choice)
                                    st.pyplot(fig_roc_train)
                                    store_figure(fig_roc_train, "ROC_Train.pdf")
                                with col_r2:
                                    fig_roc_test = plot_multi_roc(y_test, test_probs, "Validation", palette_choice)
                                    st.pyplot(fig_roc_test)
                                    store_figure(fig_roc_test, "ROC_Validation.pdf")

                            # ---------- Tab: 模型性能对比（多种图表） ----------
                            with tab_bar:
                                st.markdown("#### 模型性能对比")
                                metric_target = st.selectbox("选择要对比的指标",
                                                             ["AUC", "Accuracy", "F1_Score", "Sensitivity",
                                                              "Specificity"],
                                                             key="metric_target")
                                chart_type = st.selectbox("选择图表类型",
                                                          ["柱状图 (Bar)", "森林图 (Forest)", "散点图 (Scatter)",
                                                           "雷达图 (Radar)"],
                                                          key="chart_type")

                                if chart_type == "柱状图 (Bar)":
                                    fig = plot_metric_bars(df_test_metrics, metric_target, "Validation", palette_choice)
                                elif chart_type == "森林图 (Forest)":
                                    fig = plot_forest_style(df_test_metrics, metric_target, "Validation",
                                                            palette_choice)
                                elif chart_type == "散点图 (Scatter)":
                                    x_metric = st.selectbox("X轴指标", ["AUC", "Accuracy", "F1_Score", "Sensitivity",
                                                                        "Specificity"],
                                                            key="x_metric")
                                    y_metric = st.selectbox("Y轴指标", ["AUC", "Accuracy", "F1_Score", "Sensitivity",
                                                                        "Specificity"],
                                                            key="y_metric", index=1)
                                    fig = plot_scatter_comparison(df_test_metrics, x_metric, y_metric, "Validation",
                                                                  palette_choice)
                                elif chart_type == "雷达图 (Radar)":
                                    radar_metrics = st.multiselect("选择雷达图指标",
                                                                   ["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"],
                                                                   default=["AUC", "Accuracy", "F1_Score", "Sensitivity", "Specificity"])
                                    if len(radar_metrics) < 3:
                                        st.warning("请至少选择3个指标")
                                        fig = None
                                    else:
                                        fig = plot_radar_chart(df_test_metrics, radar_metrics, "Validation",
                                                               palette_choice)

                                if fig is not None:
                                    st.pyplot(fig)
                                    store_figure(fig, f"{chart_type}_{metric_target}.pdf")

                            # ---------- Tab: 混淆矩阵 ----------
                            with tab_cm:
                                st.markdown("#### 验证集 (Validation Set) 混淆矩阵")
                                fig_cm = plot_confusion_matrices(y_test, test_preds)
                                st.pyplot(fig_cm)
                                store_figure(fig_cm, "Confusion_Matrices.pdf")

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
                                                         palette_choice)
                            st.pyplot(fig_roc_ext)
                            store_figure(fig_roc_ext, "External_ROC.pdf")

                            # 校准曲线
                            fig_cal_ext = plot_calibration_curve(y_ext_clean, y_ext_prob, "External", palette_choice)
                            st.pyplot(fig_cal_ext)
                            store_figure(fig_cal_ext, "External_Calibration.pdf")

                            # 决策曲线
                            fig_dca_ext = plot_decision_curve(y_ext_clean, y_ext_prob, "External", palette_choice)
                            st.pyplot(fig_dca_ext)
                            store_figure(fig_dca_ext, "External_DecisionCurve.pdf")

                            # 混淆矩阵
                            fig_cm_ext = plot_confusion_matrices(y_ext_clean, {"External": y_ext_pred})
                            st.pyplot(fig_cm_ext)
                            store_figure(fig_cm_ext, "External_ConfusionMatrix.pdf")

                            st.success("外部验证完成！所有图表已存入全局下载池。")


