import io
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from openai import OpenAI

import plotly.io as pio
import plotly.graph_objects as go

# 关键修复：重置 Plotly 的默认模板，避免被 Streamlit 覆盖颜色
pio.templates.default = "plotly"

st.set_page_config(
    page_title="出口市场趋势与机会诊断平台",
    page_icon="🌏",
    layout="wide",
)


REQUIRED = ["数据年月", "商品编码", "商品名称", "贸易伙伴编码", "贸易伙伴名称", "人民币"]

st.title("🌏 出口市场趋势与机会诊断平台")
st.caption("原型版本：基于海关月度出口数据进行市场结构分析。所有指标由程序计算；AI解读模块可在后续版本接入。")

REQUIRED = ["数据年月", "商品编码", "商品名称", "贸易伙伴编码", "贸易伙伴名称", "人民币"]


def read_one_file(uploaded_file):
    """读取单个上传文件，返回 DataFrame（全部按字符串读）。"""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), dtype=str)
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), dtype=str, encoding_errors="replace")


with st.sidebar:
    st.header("数据与参数")
    uploaded_files = st.file_uploader(
        "上传海关导出文件（CSV / XLSX，可多选）",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="data_files",
    )
    st.markdown("---")
    st.markdown("**预期字段**")
    st.caption("数据年月、商品编码、商品名称、贸易伙伴编码、贸易伙伴名称、人民币")
    st.caption("请尽量保留原始表头；金额应为人民币数值。")

if not uploaded_files:
    st.info("请先上传至少一个年度数据文件。无需手动清洗或修改表头。")
    st.markdown("""
    ### 当前版本包含
    - 月度出口总额趋势
    - 目的地市场排名与份额
    - 目的地月度趋势对比
    - 市场集中度（HHI）与数据质量检查
    - 可下载的汇总结果
    """)
    st.stop()

data_list = []
for uploaded_file in uploaded_files:
    temp_df = read_one_file(uploaded_file)

    # 清理列名
    temp_df.columns = [str(col).strip() for col in temp_df.columns]

    # 兼容旧字段名
    if "贸易伙伴名称" not in temp_df.columns and "贸易伙伴名" in temp_df.columns:
        temp_df = temp_df.rename(columns={"贸易伙伴名": "贸易伙伴名称"})

    # 检查字段
    missing = [col for col in REQUIRED if col not in temp_df.columns]
    if missing:
        st.error(f"文件 {uploaded_file.name} 缺少字段：{missing}，已跳过。")
        continue

    data_list.append(temp_df)

if not data_list:
    st.error("没有可用的数据文件（全部缺少必需字段）。")
    st.stop()

raw_df = pd.concat(data_list, ignore_index=True)

# 去重
before_count = len(raw_df)
raw_df = raw_df.drop_duplicates()
duplicate_count = before_count - len(raw_df)

st.success(
    f"已成功读取 {len(data_list)} 个文件，合并后共 {len(raw_df):,} 条记录。"
)
if duplicate_count > 0:
    st.info(f"已去除 {duplicate_count} 条完全重复的记录。")

raw_df.columns = [str(c).strip() for c in raw_df.columns]
missing_cols = [c for c in REQUIRED if c not in raw_df.columns]
if missing_cols:
    st.error("缺少必要列：" + "、".join(missing_cols))
    st.write("当前文件识别到的列名：", list(raw_df.columns))
    st.stop()

df = raw_df[REQUIRED].copy()
for col in ["数据年月", "商品编码", "商品名称", "贸易伙伴编码", "贸易伙伴名称"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# Normalize YYYYMM values, while preserving invalid values for the QA panel.
df["数据年月"] = df["数据年月"].str.replace(r"\.0$", "", regex=True)
df["月份"] = pd.to_datetime(df["数据年月"], format="%Y%m", errors="coerce")
df["人民币"] = (
    df["人民币"].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("，", "", regex=False)
    .str.replace("¥", "", regex=False)
    .str.replace("￥", "", regex=False)
    .str.strip()
)
df["人民币"] = pd.to_numeric(df["人民币"], errors="coerce")
df["商品编码"] = df["商品编码"].str.replace(r"\.0$", "", regex=True)
df["贸易伙伴编码"] = df["贸易伙伴编码"].str.replace(r"\.0$", "", regex=True)

invalid_month = int(df["月份"].isna().sum())
invalid_amount = int(df["人民币"].isna().sum())
blank_partner = int((df["贸易伙伴名称"] == "").sum())

# Aggregate rows can duplicate the sum of individual partners, so exclude by default.
aggregate_pattern = r"合计|总计|全部|全球|世界"
aggregate_mask = df["贸易伙伴名称"].str.contains(aggregate_pattern, regex=True, na=False)
with st.sidebar:
    exclude_aggregates = st.checkbox("排除疑似汇总行（建议勾选）", value=True)
    if aggregate_mask.any():
        st.caption(f"检测到 {int(aggregate_mask.sum())} 行疑似汇总记录。请结合原始网站口径确认。")

if exclude_aggregates:
    df = df.loc[~aggregate_mask].copy()

df = df.dropna(subset=["月份", "人民币"])
df = df[df["贸易伙伴名称"] != ""].copy()
if df.empty:
    st.error("清洗后没有可分析的记录，请检查日期、金额及贸易伙伴字段。")
    st.stop()

# Show product selector if a file contains more than one product.
products = (
    df[["商品编码", "商品名称"]].drop_duplicates()
    .sort_values(["商品编码", "商品名称"])
)
if len(products) > 1:
    product_labels = {
        f'{r["商品编码"]} — {r["商品名称"]}': (r["商品编码"], r["商品名称"])
        for _, r in products.iterrows()
    }
    chosen_label = st.sidebar.selectbox("选择商品", list(product_labels.keys()))
    chosen_code, chosen_name = product_labels[chosen_label]
    df = df[df["商品编码"] == chosen_code]
else:
    chosen_code = products.iloc[0]["商品编码"]
    chosen_name = products.iloc[0]["商品名称"]

months = sorted(df["月份"].dropna().unique())
if not months:
    st.error("没有识别到有效月份。请确认“数据年月”格式类似 202501。")
    st.stop()

month_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
default_start = 0
default_end = len(months) - 1
with st.sidebar:
    selected_range = st.select_slider(
        "分析月份范围",
        options=month_labels,
        value=(month_labels[default_start], month_labels[default_end]),
    )
    top_n = st.slider("展示目的地数量", min_value=5, max_value=30, value=10, step=5)

start_month = pd.Timestamp(selected_range[0])
end_month = pd.Timestamp(selected_range[1])
period_df = df[(df["月份"] >= start_month) & (df["月份"] <= end_month)].copy()
if period_df.empty:
    st.warning("所选月份范围内没有数据。")
    st.stop()

# Duplicate key checks; duplicates should be reviewed rather than silently summed.
duplicate_mask = period_df.duplicated(
    subset=["月份", "商品编码", "贸易伙伴编码", "贸易伙伴名称"], keep=False
)
duplicate_rows = int(duplicate_mask.sum())

monthly = (
    period_df.groupby("月份", as_index=False)["人民币"].sum()
    .sort_values("月份")
)
market = (
    period_df.groupby(["贸易伙伴编码", "贸易伙伴名称"], as_index=False)["人民币"].sum()
    .sort_values("人民币", ascending=False)
)
total_value = float(market["人民币"].sum())
if total_value != 0:
    market["市场份额(%)"] = market["人民币"] / total_value * 100
else:
    market["市场份额(%)"] = 0.0
market["累计市场份额(%)"] = market["市场份额(%)"].cumsum()

# HHI on a 0-10,000 scale. This is descriptive concentration, not a legal competition finding.
hhi = float((market["市场份额(%)"] ** 2).sum())

months_count = int(period_df["月份"].nunique())
partners_count = int(period_df["贸易伙伴名称"].nunique())
records_count = len(period_df)

st.subheader(f"分析对象：{chosen_code} — {chosen_name}")
st.caption(f"数据范围：{selected_range[0]} 至 {selected_range[1]}｜有效记录：{records_count:,} 行")

def format_export_amount(value):
    """把元金额格式化为 亿元 / 万元 / 元，便于展示。"""
    value = float(value)
    if abs(value) >= 100_000_000:
        return f"{value / 100_000_000:,.0f} 亿元"
    elif abs(value) >= 10_000:
        return f"{value / 10_000:,.0f} 万元"
    else:
        return f"{value:,.0f} 元"

def build_trade_diagnostics(source_df):
    """
    计算出口增长贡献、增长广度、市场集中度和市场四象限。
    使用当前筛选后的 source_df，不改变原始数据。
    """

    data = source_df.copy()

    # 1. 数据清洗
    data["月份"] = pd.to_datetime(data["月份"], errors="coerce")
    data["人民币"] = pd.to_numeric(
        data["人民币"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    data = data.dropna(subset=["月份", "人民币"]).copy()
    data["月份期间"] = data["月份"].dt.to_period("M")

    if "贸易伙伴编码" not in data.columns:
        data["贸易伙伴编码"] = ""

    if "贸易伙伴名称" not in data.columns:
        data["贸易伙伴名称"] = "未知市场"

    data["贸易伙伴编码"] = (
        data["贸易伙伴编码"].fillna("").astype(str)
    )
    data["贸易伙伴名称"] = (
        data["贸易伙伴名称"].fillna("未知市场").astype(str)
    )

    if data.empty:
        return None

    # 2. 确定最近12个月和此前12个月
    latest_period = data["月份期间"].max()

    recent_periods = pd.period_range(
        end=latest_period, periods=12, freq="M"
    )
    previous_periods = pd.period_range(
        end=latest_period - 12, periods=12, freq="M"
    )

    all_periods = previous_periods.append(recent_periods)

    observed_periods = set(data["月份期间"].unique())
    missing_periods = [
        str(p) for p in all_periods
        if p not in observed_periods
    ]

    # 3. 按目的地汇总月度出口额
    group_cols = ["贸易伙伴编码", "贸易伙伴名称"]

    market_monthly = (
        data.groupby(group_cols + ["月份期间"], as_index=False)["人民币"]
        .sum()
    )

    market_pivot = market_monthly.pivot_table(
        index=group_cols,
        columns="月份期间",
        values="人民币",
        aggfunc="sum",
        fill_value=0
    )

    # 补齐最近12个月的月份
    market_pivot = market_pivot.reindex(
        columns=recent_periods, fill_value=0
    )

    recent_export = market_pivot.sum(axis=1)

    # 最近12个月的集中度
    recent_total = recent_export.sum()

    if recent_total > 0:
        recent_share = recent_export / recent_total
        recent_hhi = float((recent_share ** 2).sum())
        recent_cr1 = float(recent_share.nlargest(1).sum())
        recent_cr3 = float(recent_share.nlargest(3).sum())
        effective_markets = (
            1 / recent_hhi if recent_hhi > 0 else np.nan
        )
    else:
        recent_share = recent_export * np.nan
        recent_hhi = np.nan
        recent_cr1 = np.nan
        recent_cr3 = np.nan
        effective_markets = np.nan

    concentration = {
        "recent_total": recent_total,
        "recent_hhi": recent_hhi,
        "recent_cr1": recent_cr1,
        "recent_cr3": recent_cr3,
        "effective_markets": effective_markets,
        "recent_period_label": (
            f"{recent_periods[0]} 至 {recent_periods[-1]}"
        )
    }

    # 4. 至少有完整24个月数据，才计算同比和增长贡献
    growth_available = len(missing_periods) == 0

    if growth_available:
        market_pivot_24 = market_monthly.pivot_table(
            index=group_cols,
            columns="月份期间",
            values="人民币",
            aggfunc="sum",
            fill_value=0
        ).reindex(columns=all_periods, fill_value=0)

        previous_export = market_pivot_24[
            previous_periods
        ].sum(axis=1)

        recent_export_24 = market_pivot_24[
            recent_periods
        ].sum(axis=1)

        previous_total = previous_export.sum()
        recent_total_24 = recent_export_24.sum()

        total_change = recent_total_24 - previous_total

        market_table = pd.DataFrame({
            "此前12个月出口额": previous_export,
            "最近12个月出口额": recent_export_24
        })

        market_table["出口额变化"] = (
            market_table["最近12个月出口额"]
            - market_table["此前12个月出口额"]
        )

        market_table["同比增长率"] = np.where(
            market_table["此前12个月出口额"] > 0,
            market_table["出口额变化"]
            / market_table["此前12个月出口额"],
            np.nan
        )

        # 增长贡献率：市场出口额变化 / 整体出口额变化
        if not np.isclose(total_change, 0):
            market_table["增长贡献率"] = (
                market_table["出口额变化"] / total_change
            )
        else:
            market_table["增长贡献率"] = np.nan

        # 最近12个月市场份额
        market_table["最近12个月市场份额"] = (
            market_table["最近12个月出口额"]
            / recent_total_24
            if recent_total_24 > 0 else np.nan
        )

        # 最近12个月月度波动系数
        monthly_cv = (
            market_pivot_24[recent_periods].std(axis=1, ddof=0)
            / market_pivot_24[recent_periods].mean(axis=1).replace(0, np.nan)
        )
        market_table["月度波动系数"] = monthly_cv

        # 市场增长广度：只统计此前12个月出口额大于0的存量市场
        existing_markets = market_table[
            market_table["此前12个月出口额"] > 0
        ]

        positive_count = int(
            (existing_markets["出口额变化"] > 0).sum()
        )
        negative_count = int(
            (existing_markets["出口额变化"] < 0).sum()
        )
        unchanged_count = int(
            (existing_markets["出口额变化"] == 0).sum()
        )

        valid_count = len(existing_markets)

        growth_breadth = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "unchanged_count": unchanged_count,
            "valid_count": valid_count,
            "positive_ratio": (
                positive_count / valid_count
                if valid_count > 0 else np.nan
            ),
            "new_market_count": int(
                (
                    (market_table["此前12个月出口额"] == 0)
                    & (market_table["最近12个月出口额"] > 0)
                ).sum()
            ),
            "total_growth_rate": (
                total_change / previous_total
                if previous_total > 0 else np.nan
            ),
            "total_change": total_change
        }

        # 5. 市场四象限：以市场增长率和出口规模的中位数为界
        eligible = market_table[
            market_table["同比增长率"].notna()
            & (market_table["最近12个月出口额"] > 0)
        ].copy()

        if not eligible.empty:
            growth_median = eligible["同比增长率"].median()
            scale_median = eligible["最近12个月出口额"].median()

            def classify_market(row):
                high_growth = row["同比增长率"] >= growth_median
                high_scale = (
                    row["最近12个月出口额"] >= scale_median
                )

                if high_growth and high_scale:
                    return "重点市场"
                elif not high_growth and high_scale:
                    return "成熟市场"
                elif high_growth and not high_scale:
                    return "潜力市场"
                else:
                    return "观察市场"

            market_table["市场类型"] = market_table.apply(
                classify_market, axis=1
            )

        else:
            growth_median = np.nan
            scale_median = np.nan
            market_table["市场类型"] = "暂无法分类"

        # 新进入市场单独标记，不把它们的增长率当作无穷大
        new_mask = (
            (market_table["此前12个月出口额"] == 0)
            & (market_table["最近12个月出口额"] > 0)
        )
        market_table.loc[new_mask, "市场类型"] = "新增市场"

        # 补充目的地名称，便于展示
        market_table = market_table.reset_index()
        market_table["目的地"] = (
            market_table["贸易伙伴名称"]
        )

        # 按最近12个月出口额排序
        market_table = market_table.sort_values(
            "最近12个月出口额", ascending=False
        )

        # 此前12个月集中度，用于观察变化
        previous_share = (
            previous_export / previous_export.sum()
            if previous_export.sum() > 0
            else previous_export * np.nan
        )

        previous_hhi = (
            float((previous_share ** 2).sum())
            if previous_export.sum() > 0 else np.nan
        )

        previous_cr1 = (
            float(previous_share.nlargest(1).sum())
            if previous_export.sum() > 0 else np.nan
        )

        previous_cr3 = (
            float(previous_share.nlargest(3).sum())
            if previous_export.sum() > 0 else np.nan
        )

        concentration.update({
            "previous_hhi": previous_hhi,
            "previous_cr1": previous_cr1,
            "previous_cr3": previous_cr3
        })

    else:
        market_table = market_pivot.copy()
        market_table = market_table.reset_index()
        market_table["目的地"] = market_table["贸易伙伴名称"]
        market_table["市场类型"] = "数据不足"
        growth_breadth = None
        growth_median = np.nan
        scale_median = np.nan

    return {
        "market_table": market_table,
        "concentration": concentration,
        "growth_breadth": growth_breadth,
        "growth_available": growth_available,
        "missing_periods": missing_periods,
        "growth_median": growth_median,
        "scale_median": scale_median
    }
    

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "累计出口额",
    format_export_amount(total_value),
    help=f"完整数值：{total_value:,.2f} 元",
)
c2.metric("目的地数量", f"{partners_count:,}")
c3.metric("月度覆盖", f"{months_count} 个月")
c4.metric(
    "市场集中度 HHI",
    f"{hhi:,.0f}",
    help="按各目的地市场份额平方和计算，范围通常为 0–10,000。这里只作描述性指标。",
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "出口趋势",
    "目的地市场",
    "市场对比",
    "机会诊断",
    "出口预测",
    "AI智能诊断",
    "数据质量",
    "报告生成"
])

with tab1:
    st.markdown("#### 月度出口额")
    fig = px.line(
        monthly, x="月份", y="人民币", markers=True,
        labels={"月份": "月份", "人民币": "出口额（人民币）"},
        title="月度出口额趋势",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    monthly_show = monthly.copy()
    monthly_show["月份"] = monthly_show["月份"].dt.strftime("%Y-%m")
    monthly_show["人民币"] = monthly_show["人民币"].round(2)
    st.dataframe(monthly_show.rename(columns={"人民币": "出口额（人民币）"}), 
                column_config={
                     "出口额（人民币）": st.column_config.NumberColumn(format="%,.2f")
                },
                use_container_width=True, 
                hide_index=True)
    st.download_button(
        "下载月度出口汇总 CSV",
        data=monthly_show.to_csv(index=False).encode("utf-8-sig"),
        file_name="monthly_export_summary.csv",
        mime="text/csv",
    )
    if months_count < 36:
        st.warning(
            "当前数据不足 36 个月，暂不建议将月度趋势直接用于可靠的季节性判断。"
            "建议补充至少 3 年的连续月度数据，以便更充分地观察年度周期特征。"
        )
with tab2:
    st.markdown("#### 目的地市场排名")
    top_market = market.head(top_n).sort_values("人民币", ascending=True)
    fig = px.bar(
        top_market, x="人民币", y="贸易伙伴名称", orientation="h",
        labels={"人民币": "出口额（人民币）", "贸易伙伴名称": "国家/地区"},
        title=f"出口额排名前 {min(top_n, len(market))} 的目的地",
        hover_data={"市场份额(%)": ":.2f"},
    )
    st.plotly_chart(fig, use_container_width=True)

    display_market = market.copy()
    display_market["人民币"] = display_market["人民币"].round(2)
    display_market["市场份额(%)"] = display_market["市场份额(%)"].round(3)
    display_market["累计市场份额(%)"] = display_market["累计市场份额(%)"].round(3)
    st.dataframe(
        display_market.rename(columns={
            "贸易伙伴编码": "贸易伙伴编码",
            "贸易伙伴名称": "国家/地区",
            "人民币": "出口额（人民币）",
        }),
        column_config={
            "出口额（人民币）": st.column_config.NumberColumn(format="%,.2f")
        },
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "下载目的地市场汇总 CSV",
        data=display_market.to_csv(index=False).encode("utf-8-sig"),
        file_name="destination_market_summary.csv",
        mime="text/csv",
    )


with tab3:
    st.markdown("#### 选择目的地，比较月度变化")

    destination_names = sorted(
        period_df["贸易伙伴名称"].dropna().unique().tolist()
    )

    default_destinations = (
        market.head(min(5, len(market)))["贸易伙伴名称"].tolist()
    )

    chosen_destinations = st.multiselect(
        "选择要比较的国家/地区",
        options=destination_names,
        default=default_destinations,
    )

    if chosen_destinations:
        # 1. 汇总所选市场的月度出口额
        comp = (
            period_df[
                period_df["贸易伙伴名称"].isin(chosen_destinations)
            ]
            .groupby(["月份", "贸易伙伴名称"], as_index=False)["人民币"]
            .sum()
        )

        # 2. 月度出口额对比图
        fig = px.line(
            comp,
            x="月份",
            y="人民币",
            color="贸易伙伴名称",
            markers=True,
            labels={
                "月份": "月份",
                "人民币": "出口额（人民币）",
                "贸易伙伴名称": "国家/地区",
            },
            title="不同目的地的月度出口额对比",
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. 所选市场横向指标汇总
        st.markdown("#### 所选市场指标对比")

        total_export = period_df["人民币"].sum()

        comp_summary = (
            comp.groupby("贸易伙伴名称")
            .agg(
                累计出口额=("人民币", "sum"),
                月均出口额=("人民币", "mean"),
                月度最高出口额=("人民币", "max"),
            )
            .reset_index()
        )

        comp_summary["市场份额(%)"] = (
            comp_summary["累计出口额"] / total_export * 100
            if total_export > 0 else 0
        )

        comp_summary = comp_summary.sort_values(
            "累计出口额", ascending=False
        )

        display_comp = comp_summary.copy()

        for col in ["累计出口额", "月均出口额", "月度最高出口额"]:
            display_comp[col] = display_comp[col].round(2)

        display_comp["市场份额(%)"] = (
            display_comp["市场份额(%)"].round(2)
        )

        st.dataframe(
            display_comp.rename(columns={
                "贸易伙伴名称": "国家/地区",
                "累计出口额": "累计出口额（人民币）",
                "月均出口额": "月均出口额（人民币）",
                "月度最高出口额": "月度最高出口额（人民币）",
            }),
            column_config={
                "累计出口额（人民币）": st.column_config.NumberColumn(
                    format="%,.2f"
                ),
                "月均出口额（人民币）": st.column_config.NumberColumn(
                    format="%,.2f"
                ),
                "月度最高出口额（人民币）": st.column_config.NumberColumn(
                    format="%,.2f"
                ),
                "市场份额(%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("请选择至少一个国家/地区。")

# =====================================================
# 模块4：机会诊断
# =====================================================

with tab4:

    st.markdown("#### 市场机会与风险诊断")

    st.caption(
        "从市场结构、出口变化、增长贡献和市场潜力等角度，"
        "分析当前出口目的地的分布特征与变化情况。"
    )

    # -------------------------------------------------
    # 1. 构建目的地月度出口额矩阵
    # -------------------------------------------------

    partner_month = (
        period_df.groupby(
            ["月份", "贸易伙伴编码", "贸易伙伴名称"],
            as_index=False
        )["人民币"]
        .sum()
        .sort_values("月份")
    )

    pivot = partner_month.pivot_table(
        index="月份",
        columns="贸易伙伴编码",
        values="人民币",
        aggfunc="sum",
        fill_value=0
    ).sort_index()

    # -------------------------------------------------
    # 2. 计算目的地市场规模和市场份额
    # -------------------------------------------------

    diag = (
        period_df.groupby(
            "贸易伙伴编码",
            as_index=False
        )
        .agg(
            贸易伙伴名称=("贸易伙伴名称", "first"),
            区间出口额=("人民币", "sum")
        )
    )

    total_export = diag["区间出口额"].sum()

    if total_export > 0:
        diag["区间市场份额(%)"] = (
            diag["区间出口额"] / total_export * 100
        )
    else:
        diag["区间市场份额(%)"] = 0

    # -------------------------------------------------
    # 3. 计算首末月变化
    # -------------------------------------------------

    if len(pivot.index) >= 2:

        first_month = pivot.index[0]
        last_month = pivot.index[-1]

        first_values = pivot.iloc[0]
        last_values = pivot.iloc[-1]

        diag["首月出口额"] = (
            diag["贸易伙伴编码"]
            .map(first_values)
            .fillna(0)
        )

        diag["末月出口额"] = (
            diag["贸易伙伴编码"]
            .map(last_values)
            .fillna(0)
        )

        diag["首末月变化额"] = (
            diag["末月出口额"] - diag["首月出口额"]
        )

        diag["首末月变化率(%)"] = diag.apply(
            lambda row:
            (row["末月出口额"] / row["首月出口额"] - 1) * 100
            if row["首月出口额"] > 0
            else float("nan"),
            axis=1
        )

        # -------------------------------------------------
        # 4. 计算月度波动系数
        # -------------------------------------------------

        monthly_mean = pivot.mean(axis=0)
        monthly_std = pivot.std(axis=0, ddof=1)

        cv = (
            monthly_std /
            monthly_mean.replace(0, float("nan"))
        )

        diag["月度波动系数"] = (
            diag["贸易伙伴编码"].map(cv.to_dict())
        )

    else:

        for col in [
            "首月出口额",
            "末月出口额",
            "首末月变化额",
            "首末月变化率(%)",
            "月度波动系数"
        ]:
            diag[col] = float("nan")

    # -------------------------------------------------
    # 5. 计算各月市场集中度 HHI
    # -------------------------------------------------

    hhi_rows = []

    for month, group in partner_month.groupby("月份"):

        # 以贸易伙伴编码为单位汇总，避免同一市场重复计算
        market_values = (
            group.groupby("贸易伙伴编码")["人民币"]
            .sum()
        )

        month_total = market_values.sum()

        if month_total > 0:
            shares = market_values / month_total * 100
            month_hhi = (shares ** 2).sum()
        else:
            month_hhi = 0

        hhi_rows.append({
            "月份": month,
            "HHI": month_hhi
        })

    hhi_trend = pd.DataFrame(hhi_rows)

    # -------------------------------------------------
    # 6. 核心指标展示
    # -------------------------------------------------

    top5_share = (
        diag.nlargest(
            min(5, len(diag)),
            "区间出口额"
        )["区间市场份额(%)"].sum()
    )
    
    # 区间市场集中度 HHI，采用百分数平方口径
    hhi = (
        (diag["区间市场份额(%)"] ** 2).sum()
        if total_export > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "目的地数量",
        f"{diag['贸易伙伴名称'].nunique():,}"
    )

    col2.metric(
        "前五大市场份额",
        f"{top5_share:.2f}%"
    )

    col3.metric(
        "区间市场集中度 HHI",
        f"{hhi:,.0f}"
    )

    # -------------------------------------------------
    # 7. HHI 月度趋势图
    # -------------------------------------------------

    st.markdown("### 01 市场结构与集中度")
    st.markdown("##### 各月目的地市场集中度变化")

    if not hhi_trend.empty:

        hhi_plot = hhi_trend.copy()

        hhi_plot["月份"] = (
            hhi_plot["月份"].dt.strftime("%Y-%m")
        )

        fig_hhi = px.line(
            hhi_plot,
            x="月份",
            y="HHI",
            markers=True,
            title="各月目的地市场集中度变化",
            labels={
                "月份": "月份",
                "HHI": "市场集中度 HHI"
            }
        )

        st.plotly_chart(
            fig_hhi,
            use_container_width=True
        )

    st.caption(
        "HHI 越高，说明出口额越集中于少数目的地。"
        "该指标仅描述市场结构，不能单独证明市场风险。"
    )
    st.markdown("##### 最近12个月市场集中度")

    trade_diag = build_trade_diagnostics(period_df)
    concentration = trade_diag["concentration"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "最近12个月 CR1",
        f"{concentration['recent_cr1'] * 100:.2f}%"
    )

    c2.metric(
        "最近12个月 CR3",
        f"{concentration['recent_cr3'] * 100:.2f}%"
    )

    c3.metric(
        "有效市场数量",
        f"{concentration['effective_markets']:.1f}"
    )

    if trade_diag["growth_available"]:
        previous_hhi = concentration["previous_hhi"]
        recent_hhi = concentration["recent_hhi"]

        st.caption(
            f"最近12个月 HHI：{recent_hhi:.4f}；"
            f"此前12个月 HHI：{previous_hhi:.4f}。"
            "以上两个 HHI 均采用市场份额比例平方口径，"
            "与顶部区间 HHI 的百分数平方口径不同。"
        )
    else:
        st.info(
            "当前数据未覆盖完整的连续24个月，"
            "暂时无法比较此前12个月与最近12个月的市场集中度。"
        )

    # -------------------------------------------------
    # 8. 首末月变化分析
    # -------------------------------------------------

    st.markdown("### 02 市场规模与出口变化")
    st.markdown("##### 首末月出口变化筛查")

    if len(pivot.index) >= 2:

        first_label = pd.Timestamp(first_month).strftime("%Y-%m")
        last_label = pd.Timestamp(last_month).strftime("%Y-%m")

        st.write(
            f"比较时间：{first_label} 至 {last_label}"
        )

        valid = diag.dropna(
            subset=["首末月变化率(%)"]
        ).copy()
        
        if valid.empty:
            st.info(
                "没有首月出口额大于零的目的地，因此无法计算首末月变化率。"
            )
        else:
            left, right = st.columns(2)

        with left:

            st.markdown("**首末月变化率较高的市场**")

            increase = (
                valid.sort_values(
                    "首末月变化率(%)",
                    ascending=False
                ).head(10)
            )

            display_increase = increase[
                ["贸易伙伴名称", "首月出口额", "末月出口额", "首末月变化率(%)"]
            ].copy().round(2)

            st.dataframe(
                display_increase,
                column_config={
                    "首月出口额": st.column_config.NumberColumn(format="%,.2f"),
                    "末月出口额": st.column_config.NumberColumn(format="%,.2f"),
                    "首末月变化率(%)": st.column_config.NumberColumn(format="%.2f%%"),
                },
                use_container_width=True,
                hide_index=True,
            )

        with right:

            st.markdown("**首末月变化率较低的市场**")

            decrease = (
                valid.sort_values(
                    "首末月变化率(%)",
                    ascending=True
                ).head(10)
            )

            display_decrease = decrease[
                ["贸易伙伴名称", "首月出口额", "末月出口额", "首末月变化率(%)"]
            ].copy().round(2)

            st.dataframe(
                display_decrease,
                column_config={
                    "首月出口额": st.column_config.NumberColumn(format="%,.2f"),
                    "末月出口额": st.column_config.NumberColumn(format="%,.2f"),
                    "首末月变化率(%)": st.column_config.NumberColumn(format="%.2f%%"),
                },
                use_container_width=True,
                hide_index=True,
            )


    else:

        st.info(
            "当前筛选范围不足两个不同月份，"
            "无法计算首末月变化。"
        )

    st.caption(
        "注意：首末月变化率不是同比增长率，"
        "也不能直接解释为长期增长趋势。"
    )

    # -------------------------------------------------
    # 9. 市场规模与波动程度
    # -------------------------------------------------

    st.markdown("##### 市场规模与月度波动")

    st.caption(
        "月度波动系数 = 月度出口额标准差 / 月度出口额均值。"
        "均值接近零或零值较多时，应谨慎解释。"
    )

    show_diag = (
        diag.sort_values("区间出口额", ascending=False)
        .copy()
    )

    st.dataframe(
        show_diag,
        column_config={
            "区间出口额": st.column_config.NumberColumn(format="%,.2f"),
            "区间市场份额(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "首月出口额": st.column_config.NumberColumn(format="%,.2f"),
            "末月出口额": st.column_config.NumberColumn(format="%,.2f"),
            "首末月变化额": st.column_config.NumberColumn(format="%,.2f"),
            "首末月变化率(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "月度波动系数": st.column_config.NumberColumn(format="%.3f"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------
    # 10. 市场规模与变化率散点图
    # -------------------------------------------------

    st.markdown("##### 市场规模与首末月变化")

    if len(pivot.index) >= 2:

        scatter_data = diag.dropna(
            subset=["首末月变化率(%)"]
        ).copy()

        if not scatter_data.empty:

            fig_scatter = px.scatter(
                scatter_data,
                x="区间出口额",
                y="首末月变化率(%)",
                size="区间市场份额(%)",
                hover_name="贸易伙伴名称",
                title="目的地市场规模与首末月变化",
                labels={
                    "区间出口额": "区间出口额（人民币）",
                    "首末月变化率(%)": "首末月变化率（%）",
                    "区间市场份额(%)": "市场份额（%）"
                }
            )
            
    # 限制纵轴显示范围
    fig_scatter.update_yaxes(range=[-110, 350])

    # 极值参考线
    fig_scatter.add_hline(
        y=300,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="极值参考线（300%）",
        annotation_position="top right",
        annotation_font_color="red"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    # -------------------------------------------------
    # 11. 市场增长与机会诊断
    # -------------------------------------------------

    st.divider()
    st.markdown("### 03 市场增长与机会诊断")

    st.caption(
        "比较最近12个月与此前12个月的出口表现，"
        "识别市场增长来源、增长广度及不同规模市场的变化特征。"
    )

    # 使用统一的市场诊断函数
    # trade_diag 已在第一部分计算，这里直接复用
    growth_available = trade_diag["growth_available"]

    if not growth_available:

        st.warning(
            "当前数据未覆盖完整的连续24个月，"
            "暂时无法进行同比增长、增长贡献和四象限分析。"
        )

    else:

        growth_table = trade_diag["market_table"].copy()
        growth_breadth = trade_diag["growth_breadth"]

        st.caption(
            f"最近12个月：{trade_diag['concentration']['recent_period_label']}；"
            "增长率比较基准为此前连续12个月。"
        )

        # 1. 整体增长与市场增长广度
        st.markdown("##### 整体增长与市场增长广度")

        c1, c2, c3, c4 = st.columns(4)

        total_growth = growth_breadth["total_growth_rate"]

        c1.metric(
            "整体出口同比",
            f"{total_growth:.2%}"
            if pd.notna(total_growth) else "暂不可计算"
        )

        c2.metric(
            "增长市场数量",
            f"{growth_breadth['positive_count']} 个"
        )

        c3.metric(
            "增长市场占比",
            f"{growth_breadth['positive_ratio']:.2%}"
            if pd.notna(growth_breadth["positive_ratio"]) else "暂不可计算"
        )

        c4.metric(
            "新增市场数量",
            f"{growth_breadth['new_market_count']} 个"
        )

        st.caption(
            f"存量市场共 {growth_breadth['valid_count']} 个："
            f"其中增长 {growth_breadth['positive_count']} 个，"
            f"下降 {growth_breadth['negative_count']} 个，"
            f"持平 {growth_breadth['unchanged_count']} 个。"
        )

        # 2. 各市场增长贡献
        st.markdown("##### 各目的地市场增长贡献")

        st.caption(
            "增长贡献率 = 单个市场出口额变化 / 整体出口额变化。"
            "贡献率可能为负数，也可能超过100%，"
            "反映不同市场对整体出口额变化的贡献方向和相对大小。"
        )

        contribution_show = growth_table.sort_values(
            "增长贡献率",
            ascending=False,
            na_position="last"
        ).copy()

        # 将比例转换为百分数，仅用于展示
        contribution_show["同比增长率(%)"] = (
            contribution_show["同比增长率"] * 100
        )

        contribution_show["增长贡献率(%)"] = (
            contribution_show["增长贡献率"] * 100
        )

        contribution_show["最近12个月市场份额(%)"] = (
            contribution_show["最近12个月市场份额"] * 100
        )

        contribution_cols = [
            "目的地",
            "此前12个月出口额",
            "最近12个月出口额",
            "同比增长率(%)",
            "增长贡献率(%)",
            "最近12个月市场份额(%)",
            "市场类型"
        ]

        st.dataframe(
            contribution_show[contribution_cols],
            column_config={
                "此前12个月出口额": st.column_config.NumberColumn(
                    format="%,.2f"
                ),
                "最近12个月出口额": st.column_config.NumberColumn(
                    format="%,.2f"
                ),
                "同比增长率(%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
                "增长贡献率(%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
                "最近12个月市场份额(%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
            },
            use_container_width=True,
            hide_index=True
        )

        # 3. 市场四象限分析
        st.markdown("##### 目的地市场四象限分析")

        st.caption(
            "以市场同比增长率和最近12个月出口额的中位数为界，"
            "划分重点市场、成熟市场、潜力市场和观察市场。"
            "新增市场因此前12个月出口额为零，不计算同比增长率，单独标记。"
        )

        quadrant_df = growth_table[
            growth_table["同比增长率"].notna()
            & (growth_table["最近12个月出口额"] > 0)
        ].copy()

        if not quadrant_df.empty:

            growth_median = trade_diag["growth_median"]
            scale_median = trade_diag["scale_median"]

            fig_quadrant = px.scatter(
                quadrant_df,
                x="同比增长率",
                y="最近12个月出口额",
                color="市场类型",
                size="最近12个月市场份额",
                hover_name="目的地",
                title="目的地市场规模与同比增长率四象限分析",
                hover_data={
                    "此前12个月出口额": ":,.0f",
                    "最近12个月出口额": ":,.0f",
                    "同比增长率": ":.1%",
                    "最近12个月市场份额": ":.1%",
                },
                labels={
                    "同比增长率": "同比增长率",
                    "最近12个月出口额": "最近12个月出口额（人民币）",
                    "市场类型": "市场类型",
                    "最近12个月市场份额": "市场份额"
                }
            )

            # 限制 X 轴（同比增长率）范围。0.025 代表 2.5% (250%)
            fig_quadrant.update_xaxes(range=[-0.5, 2.5], tickformat=".0%")
            
            # 中位数分界线
            fig_quadrant.add_vline(
                x=growth_median,
                line_dash="dash",
                line_color="gray"
            )

            fig_quadrant.add_hline(
                y=scale_median,
                line_dash="dash",
                line_color="gray"
            )
            
            # 画极值参考线（用 vline 画在 x=2.0 也就是 200% 的位置）
            fig_quadrant.add_vline(
                x=2.0, 
                line_dash="dash", 
                line_color="red",
                line_width=2,
                annotation_text="极值参考线（20000%）",
                annotation_position="top right",
                annotation_font_color="red"
            )            

            fig_quadrant.update_layout(
                legend_title_text="市场类型"
            )

            st.plotly_chart(
                fig_quadrant,
                use_container_width=True
            )

        else:
            st.info("当前没有满足四象限分析条件的市场。")

        # 4. 市场分类数量
        st.markdown("##### 市场类型分布")

        type_summary = (
            growth_table["市场类型"]
            .value_counts()
            .rename_axis("市场类型")
            .reset_index(name="市场数量")
        )

        st.dataframe(
            type_summary,
            use_container_width=True,
            hide_index=True
        )

    # -------------------------------------------------
    # 12. 综合诊断与结果输出
    # -------------------------------------------------

    st.divider()
    st.markdown("### 04 综合诊断与结果输出")

    st.markdown("##### 自动生成的市场观察")

    if not diag.empty:

        # 区间出口规模最大的市场
        leader = diag.sort_values(
            "区间出口额",
            ascending=False
        ).iloc[0]

        st.write(
            f"当前分析区间内，出口额最大的目的地为"
            f" **{leader['贸易伙伴名称']}**，"
            f"出口额约为人民币 "
            f"**{leader['区间出口额']:,.0f}**，"
            f"占区间总出口额 "
            f"**{leader['区间市场份额(%)']:.2f}%**。"
        )

        st.write(
            f"前五大目的地合计占区间出口额的 "
            f"**{top5_share:.2f}%**，"
            f"区间 HHI 为 **{hhi:,.0f}**。"
        )

        if trade_diag["growth_available"]:

            # 整体增长情况
            total_growth = growth_breadth["total_growth_rate"]

            if pd.notna(total_growth):

                growth_text = (
                    f"最近12个月整体出口额较此前12个月"
                    f"{'增长' if total_growth >= 0 else '下降'}"
                    f" **{abs(total_growth):.2%}**。"
                )

                st.write(growth_text)

            st.write(
                f"在此前12个月有出口记录的存量市场中，"
                f"共有 **{growth_breadth['positive_count']}** 个市场增长，"
                f"**{growth_breadth['negative_count']}** 个市场下降，"
                f"**{growth_breadth['unchanged_count']}** 个市场持平；"
                f"另有 **{growth_breadth['new_market_count']}** 个新增市场。"
            )

            # 增长贡献最大的市场
            contribution_valid = growth_table.dropna(
                subset=["增长贡献率"]
            )

            if not contribution_valid.empty:

                top_contributor = contribution_valid.sort_values(
                    "增长贡献率",
                    ascending=False
                ).iloc[0]

                st.write(
                    f"按增长贡献率排序，"
                    f"贡献率最高的目的地为 "
                    f"**{top_contributor['目的地']}**，"
                    f"其增长贡献率为 "
                    f"**{top_contributor['增长贡献率']:.2%}**。"
                )

        else:

            st.info(
                "当前数据不足以覆盖完整的连续24个月，"
                "暂时无法生成同比增长和市场增长贡献方面的观察。"
            )

    st.caption(
        "以上内容由程序根据当前筛选数据计算生成，"
        "属于描述性统计结果，不代表因果判断或未来预测。"
    )

    # 诊断结果下载
    st.markdown("##### 分析结果下载")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "下载市场规模与变化诊断 CSV",
            data=diag.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="market_diagnosis.csv",
            mime="text/csv",
            key="download_market_diagnosis"
        )

    with col2:

        if trade_diag["growth_available"]:

            st.download_button(
                "下载市场增长与分类结果 CSV",
                data=growth_table.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="market_growth_diagnosis.csv",
                mime="text/csv",
                key="download_market_growth"
            )

        else:

            st.button(
                "市场增长数据不足，暂不可下载",
                disabled=True
            ) 
   

with tab5:
    st.subheader("出口趋势预测")

    st.markdown(
        "基于历史月度出口数据，预测未来出口趋势，"
        "并通过历史回测比较不同模型的预测效果。"
    )

    # 使用经过前面数据清洗和筛选的 period_df
    forecast_data = period_df.copy()

    # 确保出口额为数值
    forecast_data["人民币"] = pd.to_numeric(
        forecast_data["人民币"], errors="coerce"
    )
    forecast_data = forecast_data.dropna(
        subset=["人民币", "月份"]
    )

    # 统一月份格式
    forecast_data["月份"] = pd.to_datetime(
        forecast_data["月份"], errors="coerce"
    )
    forecast_data = forecast_data.dropna(subset=["月份"])
    forecast_data["月份"] = (
        forecast_data["月份"].dt.to_period("M").dt.to_timestamp()
    )

    if forecast_data.empty:
        st.warning("当前没有可用于预测的数据。")
        st.stop()

    # 选择预测对象
    market_options = ["全部市场"] + sorted(
        forecast_data["贸易伙伴名称"].dropna().unique().tolist()
    )

    selected_market = st.selectbox(
        "选择预测对象",
        market_options,
        key="forecast_market"
    )

    if selected_market == "全部市场":
        selected_data = forecast_data.copy()
        forecast_title = "第85类商品整体出口额"
    else:
        selected_data = forecast_data[
            forecast_data["贸易伙伴名称"] == selected_market
        ].copy()
        forecast_title = f"{selected_market}出口额"

    # 汇总为月度时间序列
    monthly = (
        selected_data.groupby("月份")["人民币"]
        .sum()
        .sort_index()
    )

    # 检查月份是否连续
    full_months = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS"
    )

    missing_months = full_months.difference(monthly.index)

    if len(missing_months) > 0:
        st.warning(
            f"数据中缺少 {len(missing_months)} 个月份，"
            "请检查年度文件是否完整。为避免将缺失数据误当作零出口额，"
            "当前暂不进行预测。"
        )
        st.write("缺失月份：", missing_months.strftime("%Y-%m").tolist())
        st.stop()

    monthly = monthly.reindex(full_months)
    monthly.index.name = "月份"
    monthly.name = "出口额"

    if len(monthly) < 36:
        st.warning(
            "目前有效月度数据不足36个月，"
            "建议补充历史数据后再进行预测。"
        )
        st.stop()

    st.markdown("### 历史数据概览")

    col1, col2, col3 = st.columns(3)

    col1.metric("有效月份", f"{len(monthly)}个月")
    col2.metric(
        "历史月均出口额",
        format_export_amount(monthly.mean()),
        help=f"完整数值：{monthly.mean():,.2f} 元"
    )
    col3.metric(
        "最近月份出口额",
        format_export_amount(monthly.iloc[-1]),
        help=f"完整数值：{monthly.iloc[-1]:,.2f} 元"
    )

    # 历史趋势图
    import plotly.express as px

    history_df = monthly.reset_index()
    history_df.columns = ["月份", "出口额"]

    fig_history = px.line(
        history_df,
        x="月份",
        y="出口额",
        title=f"{forecast_title}历史月度趋势",
        markers=True
    )
    fig_history.update_layout(
        xaxis_title="月份",
        yaxis_title="出口额（人民币）",
        hovermode="x unified"
    )
    st.plotly_chart(fig_history, use_container_width=True)

    st.divider()

    st.markdown("### 模型回测与比较")

    st.caption(
        "采用最后12个月作为测试集，"
        "使用此前月份训练模型，比较预测值与实际值。"
    )

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # 留出最后12个月进行回测
        train = monthly.iloc[:-12]
        test = monthly.iloc[-12:]

        # 模型一：季节性朴素预测
        # 预测值等于上一年对应月份的实际值
        seasonal_naive_pred = train.iloc[-12:].values

        # 模型二：Holt-Winters指数平滑
        ets_model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        )

        ets_fit = ets_model.fit(optimized=True)
        ets_pred = ets_fit.forecast(12)

        # 计算评价指标
        import numpy as np

        def calculate_metrics(actual, predicted):
            actual = np.asarray(actual, dtype=float)
            predicted = np.asarray(predicted, dtype=float)

            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            # 计算 WAPE：绝对误差之和 / 实际值之和
            # 判空处理：如果实际值之和为 0，则返回 NaN 防止除零报错
            sum_actual = np.sum(np.abs(actual))
            wape = np.sum(np.abs(actual - predicted)) / sum_actual * 100 if sum_actual != 0 else np.nan
            return mae, rmse, wape

        naive_mae, naive_rmse, naive_wape = calculate_metrics(
            test.values, seasonal_naive_pred
        )

        ets_mae, ets_rmse, ets_wape = calculate_metrics(
            test.values, ets_pred.values
        )

        metrics_df = pd.DataFrame({
            "模型": ["季节性朴素模型", "Holt-Winters指数平滑"],
            "MAE（元）": [naive_mae, ets_mae],
            "RMSE（元）": [naive_rmse, ets_rmse],
            "WAPE(%)": [naive_wape, ets_wape]
        })

        st.dataframe(
            metrics_df,
            column_config={
                "MAE（元）": st.column_config.NumberColumn(format="%,.2f"),
                "RMSE（元）": st.column_config.NumberColumn(format="%,.2f"),
                "WAPE（%）": st.column_config.NumberColumn(format="%.2f%%"), 
            },
            use_container_width=True,
            hide_index=True
        )

        # 回测结果可视化
        backtest_df = pd.DataFrame({
            "月份": test.index,
            "实际值": test.values,
            "季节性朴素模型": seasonal_naive_pred,
            "Holt-Winters模型": ets_pred.values
        })

        fig_test = px.line(
            backtest_df,
            x="月份",
            y=["实际值", "季节性朴素模型", "Holt-Winters模型"],
            title="最近12个月预测回测"
        )

        fig_test.update_layout(
            xaxis_title="月份",
            yaxis_title="出口额（人民币）",
            hovermode="x unified"
        )

        st.plotly_chart(fig_test, use_container_width=True)

        # 根据RMSE选择回测表现更好的模型
        if ets_rmse <= naive_rmse:
            best_model = "Holt-Winters指数平滑"
            st.success(
                "在本次历史回测中，Holt-Winters模型的RMSE较低。"
            )
        else:
            best_model = "季节性朴素模型"
            st.info(
                "在本次历史回测中，季节性朴素模型的RMSE较低。"
            )

        st.caption(
            "注意：回测结果只反映当前测试区间的表现，"
            "不代表未来预测一定具有相同精度。"
        )

        st.divider()

        st.markdown("### 未来出口额预测")

        forecast_period = st.slider(
            "预测未来月份数",
            min_value=1,
            max_value=6,
            value=3,
            key="forecast_period"
        )

        # 使用全部历史数据重新拟合两个模型
        final_ets = ExponentialSmoothing(
            monthly,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit(optimized=True)

        ets_future = final_ets.forecast(forecast_period)

        naive_future = monthly.iloc[-12:].values[
            :forecast_period
        ]

        future_index = pd.date_range(
            start=monthly.index[-1] + pd.offsets.MonthBegin(1),
            periods=forecast_period,
            freq="MS"
        )

        # 采用回测中RMSE较低的模型作为展示模型
        if ets_rmse <= naive_rmse:
            future_values = ets_future.values
            chosen_model = "Holt-Winters指数平滑"
        else:
            future_values = naive_future
            chosen_model = "季节性朴素模型"

        future_df = pd.DataFrame({
            "月份": future_index,
            "预测出口额": future_values
        })

        st.write(f"当前采用模型：**{chosen_model}**")

        st.dataframe(
            future_df,
            column_config={
                "预测出口额": st.column_config.NumberColumn(format="%,.2f"),
            },
            use_container_width=True,
            hide_index=True
        )

        # 历史数据与未来预测图
        chart_history = pd.DataFrame({
            "月份": monthly.index,
            "出口额": monthly.values,
            "类型": "历史实际值"
        })

        chart_future = pd.DataFrame({
            "月份": future_index,
            "出口额": future_values,
            "类型": "未来预测值"
        })

        chart_all = pd.concat(
            [chart_history, chart_future],
            ignore_index=True
        )

        fig_future = px.line(
            chart_all,
            x="月份",
            y="出口额",
            color="类型",
            markers=True,
            title=f"{forecast_title}未来{forecast_period}个月预测"
        )

        fig_future.update_layout(
            xaxis_title="月份",
            yaxis_title="出口额（人民币）",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_future,
            use_container_width=True
        )

        # 下载预测结果
        export_forecast = future_df.copy()
        export_forecast["预测模型"] = chosen_model

        st.download_button(
            "下载预测结果 CSV",
            data=export_forecast.to_csv(
                index=False, encoding="utf-8-sig"
            ),
            file_name="出口趋势预测结果.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"预测模型运行失败：{e}")
        st.info(
            "请检查 statsmodels 是否安装，以及历史数据是否完整。"
        )


with tab6:
    st.subheader("AI 智能市场诊断")

    st.markdown(
        "基于海关出口数据及平台计算指标，"
        "利用 DeepSeek 生成市场分析与风险提示。"
    )

    import json

    # 读取当前分析数据
    ai_data = period_df.copy()

    ai_data["人民币"] = pd.to_numeric(
        ai_data["人民币"], errors="coerce"
    )
    ai_data["月份"] = pd.to_datetime(
        ai_data["月份"], errors="coerce"
    )

    ai_data = ai_data.dropna(
        subset=[
            "人民币",
            "月份",
            "贸易伙伴编码",
            "贸易伙伴名称"
        ]
    )

    ai_data["月份"] = (
        ai_data["月份"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if ai_data.empty:
        st.warning("当前没有可用于 AI 分析的数据。")
    else:
        # 统一计算月度数据
        monthly_total = (
            ai_data.groupby("月份")["人民币"]
            .sum()
            .sort_index()
        )

        latest_month = monthly_total.index.max()
        recent_start = latest_month - pd.DateOffset(months=11)
        previous_end = recent_start - pd.DateOffset(months=1)
        previous_start = previous_end - pd.DateOffset(months=11)

        recent_months = pd.date_range(
            recent_start, latest_month, freq="MS"
        )
        previous_months = pd.date_range(
            previous_start, previous_end, freq="MS"
        )

        # 检查最近24个月是否完整
        available_months = set(monthly_total.index)
        required_months = (
            set(recent_months) | set(previous_months)
        )

        if not required_months.issubset(available_months):
            st.warning(
                "最近24个月的数据月份不完整。"
                "为避免同比结果失真，请先检查上传的数据文件。"
            )
        else:
            # 计算整体市场指标
            recent_total = monthly_total.reindex(
                recent_months
            ).sum()

            previous_total = monthly_total.reindex(
                previous_months
            ).sum()

            if previous_total > 0:
                overall_growth = (
                    recent_total - previous_total
                ) / previous_total * 100
            else:
                overall_growth = None

            monthly_cv = (
                monthly_total.reindex(recent_months).std(ddof=0)
                / monthly_total.reindex(recent_months).mean()
                if monthly_total.reindex(recent_months).mean() > 0
                else None
            )

            # 计算各市场最近12个月和此前12个月出口额
            market_monthly = (
                ai_data.groupby(
                    [
                        "贸易伙伴编码",
                        "贸易伙伴名称",
                        "月份"
                    ]
                )["人民币"]
                .sum()
                .reset_index()
            )

            market_results = []

            for (code, name), group in market_monthly.groupby(
                ["贸易伙伴编码", "贸易伙伴名称"]
            ):
                series = (
                    group.set_index("月份")["人民币"]
                    .reindex(
                        list(previous_months) + list(recent_months),
                        fill_value=0
                    )
                )

                previous_value = series.iloc[:12].sum()
                recent_series = series.iloc[12:]
                recent_value = recent_series.sum()

                growth = (
                    (recent_value - previous_value)
                    / previous_value * 100
                    if previous_value > 0
                    else None
                )

                share = (
                    recent_value / recent_total * 100
                    if recent_total > 0
                    else None
                )

                mean_value = recent_series.mean()
                cv = (
                    recent_series.std(ddof=0) / mean_value
                    if mean_value > 0
                    else None
                )

                market_results.append({
                    "贸易伙伴编码": str(code),
                    "贸易伙伴名称": str(name),
                    "最近12个月出口额": float(recent_value),
                    "此前12个月出口额": float(previous_value),
                    "同比增长率(%)": (
                        round(float(growth), 2)
                        if growth is not None else None
                    ),
                    "市场份额(%)": (
                        round(float(share), 2)
                        if share is not None else None
                    ),
                    "月度波动系数": (
                        round(float(cv), 3)
                        if cv is not None else None
                    )
                })

            market_df_ai = pd.DataFrame(market_results)

            market_df_ai = market_df_ai[
                market_df_ai["最近12个月出口额"] > 0
            ].copy()

            # HHI：市场份额平方和
            shares = market_df_ai["市场份额(%)"].dropna()
            hhi_value = float((shares ** 2).sum())

            # 主要市场排名
            top_markets = (
                market_df_ai.sort_values(
                    "最近12个月出口额",
                    ascending=False
                )
                .head(10)
                .to_dict(orient="records")
            )

            market_names = sorted(
                market_df_ai["贸易伙伴名称"].unique().tolist()
            )

            analysis_type = st.radio(
                "选择分析范围",
                ["整体市场分析", "重点市场分析"],
                horizontal=True,
                key="ai_analysis_type"
            )

            selected_market = None

            if analysis_type == "重点市场分析":
                selected_market = st.selectbox(
                    "选择贸易伙伴",
                    market_names,
                    key="ai_selected_market"
                )

            st.markdown("### 本次分析的数据范围")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "最近12个月出口额",
                format_export_amount(recent_total),
                help=f"完整数值：{recent_total:,.2f} 元",
            )

            col2.metric(
                "最近12个月同比增长",
                (
                    f"{overall_growth:.2f}%"
                    if overall_growth is not None
                    else "无法计算"
                )
            )

            col3.metric(
                "市场集中度 HHI",
                f"{hhi_value:.1f}"
            )

            st.caption(
                f"最近12个月：{recent_start:%Y-%m} 至 "
                f"{latest_month:%Y-%m}；"
                f"同比基准：{previous_start:%Y-%m} 至 "
                f"{previous_end:%Y-%m}"
            )

            # 构造传给模型的数据
            analysis_payload = {
                "商品类别": "第85类：电机、电气设备及其零件",
                "数据范围": {
                    "最近12个月": (
                        f"{recent_start:%Y-%m} 至 "
                        f"{latest_month:%Y-%m}"
                    ),
                    "同比基准期": (
                        f"{previous_start:%Y-%m} 至 "
                        f"{previous_end:%Y-%m}"
                    )
                },
                "整体指标": {
                    "最近12个月出口额_人民币": round(
                        float(recent_total), 2
                    ),
                    "此前12个月出口额_人民币": round(
                        float(previous_total), 2
                    ),
                    "同比增长率_pct": (
                        round(float(overall_growth), 2)
                        if overall_growth is not None else None
                    ),
                    "月度出口额波动系数": (
                        round(float(monthly_cv), 3)
                        if monthly_cv is not None else None
                    ),
                    "HHI市场集中度": round(hhi_value, 2),
                    "前五大市场份额_pct": round(
                        float(
                            market_df_ai.nlargest(
                                5, "最近12个月出口额"
                            )["市场份额(%)"].sum()
                        ), 2
                    )
                },
                "主要出口市场_按最近12个月出口额排序": top_markets
            }

            if selected_market is not None:
                selected_row = market_df_ai[
                    market_df_ai["贸易伙伴名称"] == selected_market
                ].iloc[0]

                analysis_payload["重点市场"] = (
                    selected_row.to_dict()
                )

            st.markdown("### 生成 AI 分析报告")

            st.caption(
                "模型将根据上述计算结果生成分析，不会重新计算或修改原始指标。"
            )

            if st.button(
                "生成 AI 市场诊断报告",
                type="primary",
                key="generate_ai_report"
            ):
                try:
                    client = OpenAI(
                        api_key=st.secrets["DEEPSEEK_API_KEY"],
                        base_url="https://api.deepseek.com"
                    )

                    if analysis_type == "整体市场分析":
                        task_instruction = (
                            "请对第85类商品的整体出口市场进行分析。"
                            "重点分析整体出口规模、同比变化、市场集中度、"
                            "主要目的地市场，以及增长机会和风险。"
                        )
                    else:
                        task_instruction = (
                            f"请重点分析贸易伙伴“{selected_market}”。"
                            "结合其出口规模、同比增长率、市场份额和波动系数，"
                            "判断其近期表现，并提出审慎的市场观察建议。"
                        )

                    system_prompt = """
你是一名专业的国际贸易数据分析师。
请严格遵守以下规则：

1. 只能使用用户提供的数据，不得编造数字、国家排名或事实。
2. 不得把相关关系解释为因果关系。
3. 如果某项指标为null，必须说明该指标无法计算，不得自行补齐。
4. 必须明确分析的时间区间。
5. 区分数据事实、分析判断和建议。
6. 不得仅凭短期增长就断言市场具有长期潜力。
7. 不得将第85类商品的整体数据解释为某个具体细分产品的数据。
8. 使用清晰、专业、自然的中文。
9. 报告分为：市场概况、增长表现、风险提示、市场建议四个部分。
10. 建议必须能追溯到提供的指标，不得提出缺乏数据依据的具体市场事实。
"""

                    user_prompt = (
                        task_instruction
                        + "\n\n以下是由 Python 计算得到的数据：\n"
                        + json.dumps(
                            analysis_payload,
                            ensure_ascii=False,
                            allow_nan=False
                        )
                    )

                    response = client.chat.completions.create(
                        model="deepseek-flash",
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        max_tokens=1800,
                        temperature=0.3,
                        extra_body={
                            "thinking": {
                                "type": "disabled"
                            }
                        }
                    )

                    report = response.choices[0].message.content

                    st.session_state["ai_market_report"] = report
                    st.session_state["ai_market_report_scope"] = (
                        analysis_type
                    )
                    st.session_state["ai_market_report_market"] = (
                        selected_market or "整体市场"
                    )

                except Exception as e:
                    st.error(f"AI 分析生成失败：{e}")

            # 显示已生成的报告
            if st.session_state.get("ai_market_report"):
                st.divider()
                st.markdown("### AI 市场诊断报告")

                st.caption(
                    "分析范围："
                    + st.session_state.get(
                        "ai_market_report_market", "整体市场"
                    )
                )

                report = st.session_state["ai_market_report"]

                st.markdown(report)

                st.download_button(
                    "下载 AI 分析报告",
                    data=report,
                    file_name="AI市场诊断报告.md",
                    mime="text/markdown",
                    key="download_ai_report"
                )

with tab7:
    st.markdown("#### 数据质量检查")
    qa = [
        ("原始记录数", len(raw_df)),
        ("清洗后有效记录数", len(df)),
        ("所选月份范围记录数", len(period_df)),
        ("无法识别的月份", invalid_month),
        ("无法转换为数值的出口额", invalid_amount),
        ("贸易伙伴名为空的记录（原始数据）", blank_partner),
        ("疑似汇总行", int(aggregate_mask.sum())),
        ("重复业务键涉及的记录数（当前范围）", duplicate_rows),
        ("月份数量（当前范围）", months_count),
        ("目的地数量（当前范围）", partners_count),
    ]
    st.dataframe(pd.DataFrame(qa, columns=["检查项", "结果"]), use_container_width=True, hide_index=True)
    if duplicate_rows:
        st.warning("发现重复的“月份 + 商品编码 + 贸易伙伴编码 + 贸易伙伴名”记录。请确认它们是可加总的拆分项，还是重复导出导致的重复记录。")
    else:
        st.success("当前筛选范围内未发现重复业务键。")
    if aggregate_mask.any():
        st.caption("系统按贸易伙伴名中的“合计、总计、全部、全球、世界”等关键词识别疑似汇总行。请以原始数据字典和网站统计口径为准，不要盲目排除或保留。")

st.markdown("---")
st.caption("口径说明：当前总出口额为所选范围内各贸易伙伴记录的加总。若源文件含有汇总行、特殊贸易伙伴或统计口径差异，需先核验后再解释。HHI 为描述性市场集中度指标，不直接等同于市场风险。")

# DeepSeek API 连接测试
st.divider()
st.subheader("DeepSeek API 测试")

if st.button("测试 DeepSeek API 连接"):
    try:
        client = OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个简洁的中文数据分析助手。"
                },
                {
                    "role": "user",
                    "content": "请用一句话介绍出口数据分析的价值。"
                }
            ],
            max_tokens=150,
            temperature=0.3,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        answer = response.choices[0].message.content

        st.success("DeepSeek API 连接成功！")
        st.write(answer)

    except Exception as e:
        st.error(f"API 调用失败：{e}")


with tab8:
    st.subheader("综合出口市场分析报告")

    st.markdown(
        "整合出口趋势、市场结构、机会诊断、预测结果和AI分析，"
        "自动生成可下载的综合报告。"
    )

    import html
    import numpy as np
    import plotly.express as px
    import plotly.io as pio
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    report_data = period_df.copy()

    report_data["人民币"] = pd.to_numeric(
        report_data["人民币"], errors="coerce"
    )
    report_data["月份"] = pd.to_datetime(
        report_data["月份"], errors="coerce"
    )

    report_data = report_data.dropna(
        subset=[
            "人民币",
            "月份",
            "贸易伙伴编码",
            "贸易伙伴名称"
        ]
    )

    report_data["月份"] = (
        report_data["月份"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if report_data.empty:
        st.warning("当前没有可用于生成报告的数据。")
    else:
        # =========================
        # 1. 整体出口数据
        # =========================

        monthly_export = (
            report_data.groupby("月份")["人民币"]
            .sum()
            .sort_index()
        )

        full_months = pd.date_range(
            monthly_export.index.min(),
            monthly_export.index.max(),
            freq="MS"
        )

        missing_months = full_months.difference(
            monthly_export.index
        )

        monthly_export = monthly_export.reindex(full_months)

        latest_month = monthly_export.index.max()
        first_month = monthly_export.index.min()

        total_export = monthly_export.sum()
        monthly_average = monthly_export.mean()

        st.markdown("### 报告数据概况")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "数据覆盖月份",
            f"{len(monthly_export)}个月"
        )

        col2.metric(
            "累计出口额",
            format_export_amount(total_export),
            help=f"完整数值：{total_export:,.2f} 元"
        )

        col3.metric(
            "月均出口额",
            format_export_amount(monthly_average),
            help=f"完整数值：{monthly_average:,.2f} 元"
        )

        col4.metric(
            "最新数据月份",
            latest_month.strftime("%Y-%m")
        )

        if len(missing_months) > 0:
            st.warning(
                "检测到缺失月份，报告会标明数据不完整；"
                "涉及连续月份的同比和预测分析将谨慎处理。"
            )

        # =========================
        # 2. 最近12个月同比
        # =========================

        recent_start = latest_month - pd.DateOffset(months=11)
        previous_end = recent_start - pd.DateOffset(months=1)
        previous_start = previous_end - pd.DateOffset(months=11)

        recent_months = pd.date_range(
            recent_start, latest_month, freq="MS"
        )

        previous_months = pd.date_range(
            previous_start, previous_end, freq="MS"
        )

        required_months = set(recent_months).union(
            set(previous_months)
        )

        available_months = set(monthly_export.index)

        has_complete_24m = (
            required_months.issubset(available_months)
            and len(missing_months) == 0
        )

        if has_complete_24m:
            recent_total = monthly_export.reindex(
                recent_months
            ).sum()

            previous_total = monthly_export.reindex(
                previous_months
            ).sum()

            overall_growth = (
                (recent_total - previous_total)
                / previous_total * 100
                if previous_total > 0
                else None
            )
        else:
            recent_total = None
            previous_total = None
            overall_growth = None

        # =========================
        # 3. 市场结构和市场指标
        # =========================

        market_monthly = (
            report_data.groupby(
                [
                    "贸易伙伴编码",
                    "贸易伙伴名称",
                    "月份"
                ]
            )["人民币"]
            .sum()
            .reset_index()
        )

        market_total = (
            market_monthly.groupby(
                ["贸易伙伴编码", "贸易伙伴名称"]
            )["人民币"]
            .sum()
            .reset_index(name="累计出口额")
        )

        market_total = market_total.sort_values(
            "累计出口额", ascending=False
        )

        total_for_share = market_total["累计出口额"].sum()

        market_total["累计市场份额(%)"] = (
            market_total["累计出口额"]
            / total_for_share * 100
            if total_for_share > 0
            else 0
        )

        top5_share = market_total.head(5)[
            "累计市场份额(%)"
        ].sum()

        hhi_total = (
            market_total["累计市场份额(%)"] ** 2
        ).sum()

        # 最近12个月市场评价
        opportunity_rows = []

        if has_complete_24m:
            recent_market = market_monthly[
                market_monthly["月份"].isin(recent_months)
            ]

            previous_market = market_monthly[
                market_monthly["月份"].isin(previous_months)
            ]

            recent_by_market = (
                recent_market.groupby(
                    ["贸易伙伴编码", "贸易伙伴名称"]
                )["人民币"].sum()
            )

            previous_by_market = (
                previous_market.groupby(
                    ["贸易伙伴编码", "贸易伙伴名称"]
                )["人民币"].sum()
            )

            for _, row in market_total.iterrows():
                key = (
                    row["贸易伙伴编码"],
                    row["贸易伙伴名称"]
                )

                recent_value = float(
                    recent_by_market.get(key, 0)
                )

                previous_value = float(
                    previous_by_market.get(key, 0)
                )

                growth = (
                    (recent_value - previous_value)
                    / previous_value * 100
                    if previous_value > 0
                    else np.nan
                )

                recent_series = (
                    recent_market[
                        (
                            recent_market["贸易伙伴编码"]
                            == key[0]
                        )
                        & (
                            recent_market["贸易伙伴名称"]
                            == key[1]
                        )
                    ]
                    .set_index("月份")["人民币"]
                    .reindex(recent_months, fill_value=0)
                )

                mean_value = recent_series.mean()

                cv = (
                    recent_series.std(ddof=0) / mean_value
                    if mean_value > 0
                    else np.nan
                )

                share = (
                    recent_value / recent_total * 100
                    if recent_total and recent_total > 0
                    else np.nan
                )

                opportunity_rows.append({
                    "贸易伙伴编码": key[0],
                    "贸易伙伴名称": key[1],
                    "最近12个月出口额": recent_value,
                    "同比增长率(%)": growth,
                    "市场份额(%)": share,
                    "波动系数": cv
                })

        opportunity_report = pd.DataFrame(
            opportunity_rows
        )

        if not opportunity_report.empty:
            opportunity_report = opportunity_report[
                opportunity_report["最近12个月出口额"] > 0
            ].copy()

            opportunity_report["增长得分"] = (
                opportunity_report["同比增长率(%)"]
                .rank(pct=True)
                .fillna(0.5) * 100
            )

            opportunity_report["规模得分"] = (
                opportunity_report["市场份额(%)"]
                .rank(pct=True)
                .fillna(0.5) * 100
            )

            opportunity_report["稳定性得分"] = (
                opportunity_report["波动系数"]
                .rank(ascending=False, pct=True)
                .fillna(0.5) * 100
            )

            opportunity_report["综合得分"] = (
                opportunity_report["增长得分"] * 0.40
                + opportunity_report["规模得分"] * 0.35
                + opportunity_report["稳定性得分"] * 0.25
            )

            opportunity_report = opportunity_report.sort_values(
                "综合得分", ascending=False
            )

            opportunity_report.insert(
                0,
                "排名",
                range(1, len(opportunity_report) + 1)
            )

        # =========================
        # 4. 预测模型
        # =========================

        st.markdown("### 预测设置")

        forecast_horizon = st.slider(
            "报告中的未来预测月份数",
            min_value=1,
            max_value=6,
            value=3,
            key="report_forecast_horizon"
        )

        forecast_result = None
        forecast_model_name = "暂不可预测"
        backtest_result = None

        if len(monthly_export) >= 36 and len(missing_months) == 0:
            try:
                train = monthly_export.iloc[:-12]
                test = monthly_export.iloc[-12:]

                naive_pred = train.iloc[-12:].values

                ets_fit = ExponentialSmoothing(
                    train,
                    trend="add",
                    damped_trend=True,
                    seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated"
                ).fit(optimized=True)

                ets_pred = ets_fit.forecast(12)

                naive_rmse = np.sqrt(
                    np.mean(
                        (test.values - naive_pred) ** 2
                    )
                )

                ets_rmse = np.sqrt(
                    np.mean(
                        (test.values - ets_pred.values) ** 2
                    )
                )

                if ets_rmse <= naive_rmse:
                    forecast_model_name = "Holt-Winters指数平滑"
                    final_fit = ExponentialSmoothing(
                        monthly_export,
                        trend="add",
                        damped_trend=True,
                        seasonal="add",
                        seasonal_periods=12,
                        initialization_method="estimated"
                    ).fit(optimized=True)

                    forecast_values = final_fit.forecast(
                        forecast_horizon
                    ).values
                else:
                    forecast_model_name = "季节性朴素模型"
                    forecast_values = monthly_export.iloc[-12:].values[
                        :forecast_horizon
                    ]

                forecast_index = pd.date_range(
                    latest_month + pd.offsets.MonthBegin(1),
                    periods=forecast_horizon,
                    freq="MS"
                )

                forecast_result = pd.DataFrame({
                    "月份": forecast_index,
                    "预测出口额": forecast_values
                })

                backtest_result = {
                    "季节性朴素模型 RMSE": float(naive_rmse),
                    "Holt-Winters模型 RMSE": float(ets_rmse)
                }

            except Exception as e:
                st.warning(f"预测结果未能生成：{e}")
        else:
            st.info(
                "连续完整的月度数据不足36个月，"
                "本次报告不包含未来预测。"
            )

        # =========================
        # 5. AI报告
        # =========================

        ai_report = st.session_state.get(
            "ai_market_report", ""
        )

        st.markdown("### AI诊断报告状态")

        if ai_report:
            st.success("已读取此前生成的 AI 市场诊断报告。")
        else:
            st.warning(
                "尚未生成 AI 市场诊断报告。"
                "请先进入“AI智能诊断”页面生成报告，"
                "再返回本页面生成综合报告。"
            )

        # =========================
        # 6. 生成图表
        # =========================

        trend_df = monthly_export.reset_index()
        trend_df.columns = ["月份", "出口额"]

        if forecast_result is not None:
            trend_plot_df = pd.concat([
                pd.DataFrame({
                    "月份": monthly_export.index,
                    "出口额": monthly_export.values,
                    "类型": "历史实际值"
                }),
                pd.DataFrame({
                    "月份": forecast_result["月份"],
                    "出口额": forecast_result["预测出口额"],
                    "类型": "未来预测值"
                })
            ], ignore_index=True)
        else:
            trend_plot_df = pd.DataFrame({
                "月份": monthly_export.index,
                "出口额": monthly_export.values,
                "类型": "历史实际值"
            })

        trend_fig = px.line(
            trend_plot_df,
            x="月份",
            y="出口额",
            color="类型",
            markers=True,
            title="月度出口趋势及未来预测"
        )

        trend_fig.update_layout(
            xaxis_title="月份",
            yaxis_title="出口额（人民币）",
            hovermode="x unified"
        )

        top_market_chart = market_total.head(10).copy()

        market_fig = px.bar(
            top_market_chart.sort_values("累计出口额"),
            x="累计出口额",
            y="贸易伙伴名称",
            orientation="h",
            title="累计出口额前十位市场"
        )

        market_fig.update_layout(
            xaxis_title="累计出口额（人民币）",
            yaxis_title="贸易伙伴"
        )

        st.markdown("### 报告预览")

        st.plotly_chart(
            trend_fig,
            use_container_width=True
        )

        st.plotly_chart(
            market_fig,
            use_container_width=True
        )

        # =========================
        # 7. 组织报告内容
        # =========================

        st.markdown("### 报告内容")

        if has_complete_24m:
            growth_text = f"{overall_growth:.2f}%"
            growth_sentence = (
                f"最近12个月出口额为 {recent_total:,.2f} 元，"
                f"相较此前12个月同比变化 {growth_text}。"
            )
        else:
            growth_text = "无法计算"
            growth_sentence = (
                "由于最近24个月数据不完整，"
                "本报告不提供整体出口额同比增长判断。"
            )

        if overall_growth is not None:
            if overall_growth > 0:
                trend_judgement = "最近12个月出口额同比增长。"
            elif overall_growth < 0:
                trend_judgement = "最近12个月出口额同比下降。"
            else:
                trend_judgement = "最近12个月出口额同比基本持平。"
        else:
            trend_judgement = "当前无法判断最近12个月的同比趋势。"

        top_market_names = "、".join(
            market_total.head(5)["贸易伙伴名称"].astype(str).tolist()
        )

        report_summary = f"""
        本报告分析中国第85类商品的出口市场表现。
        当前数据覆盖 {first_month:%Y年%m月} 至 {latest_month:%Y年%m月}，
        共包含 {len(monthly_export)} 个月度记录。

        数据期间累计出口额为 {total_export:,.2f} 元，
        月均出口额为 {monthly_average:,.2f} 元。
        {growth_sentence}

        {trend_judgement}

        按当前数据期间累计出口额计算，
        主要出口市场包括：{top_market_names}。
        前五大市场累计份额为 {top5_share:.2f}%；
        市场集中度 HHI 为 {hhi_total:.2f}。
        """

        st.markdown("#### 一、数据概况与整体趋势")
        st.write(report_summary)

        st.markdown("#### 二、主要出口市场")

        st.dataframe(
            market_total.head(10).style.format({
                "累计出口额": "{:,.2f}",
                "累计市场份额(%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### 三、市场机会评价")

        if not opportunity_report.empty:
            st.dataframe(
                opportunity_report[
                    [
                        "排名",
                        "贸易伙伴名称",
                        "最近12个月出口额",
                        "同比增长率(%)",
                        "市场份额(%)",
                        "波动系数",
                        "综合得分"
                    ]
                ].head(10).style.format({
                    "最近12个月出口额": "{:,.2f}",
                    "同比增长率(%)": "{:.2f}%",
                    "市场份额(%)": "{:.2f}%",
                    "波动系数": "{:.3f}",
                    "综合得分": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("当前数据不足以生成最近12个月市场机会评价。")

        st.markdown("#### 四、出口趋势预测")

        if forecast_result is not None:
            st.write(
                f"本次采用模型：{forecast_model_name}。"
                "模型根据最近12个月历史回测的RMSE进行选择。"
            )

            st.dataframe(
                forecast_result.style.format({
                    "预测出口额": "{:,.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.write(
                "历史回测RMSE："
                f"季节性朴素模型 {backtest_result['季节性朴素模型 RMSE']:,.2f} 元；"
                f"Holt-Winters模型 {backtest_result['Holt-Winters模型 RMSE']:,.2f} 元。"
            )
        else:
            st.info("当前没有可用的预测结果。")

        st.markdown("#### 五、AI智能市场诊断")

        if ai_report:
            st.markdown(ai_report)
        else:
            st.info(
                "尚无AI诊断内容。生成AI诊断报告后，"
                "可以重新生成本综合报告。"
            )

        st.markdown("#### 六、结论与使用说明")

        conclusion_text = (
            f"本次分析覆盖第85类商品，"
            f"数据期间为 {first_month:%Y年%m月} 至 "
            f"{latest_month:%Y年%m月}。"
            f"累计出口额为 {total_export:,.2f} 元，"
            f"前五大市场份额为 {top5_share:.2f}%，"
            f"HHI为 {hhi_total:.2f}。"
            f"{trend_judgement}"
            "市场机会综合得分是依据本平台设定的相对评分规则计算，"
            "仅用于当前样本内的市场比较，不代表市场成功概率。"
            "预测结果基于历史月度数据和回测表现，"
            "不构成对未来出口额的保证。"
        )

        st.write(conclusion_text)

        # =========================
        # 8. 生成可下载HTML
        # =========================

        trend_html = pio.to_html(
            trend_fig,
            full_html=False,
            include_plotlyjs=True
        )

        market_html = pio.to_html(
            market_fig,
            full_html=False,
            include_plotlyjs=False
        )

        market_html_table = market_total.head(10).to_html(
            index=False,
            border=0,
            classes="report-table",
            float_format=lambda x: f"{x:,.2f}"
        )

        if not opportunity_report.empty:
            opportunity_html_table = opportunity_report[
                [
                    "排名",
                    "贸易伙伴名称",
                    "最近12个月出口额",
                    "同比增长率(%)",
                    "市场份额(%)",
                    "波动系数",
                    "综合得分"
                ]
            ].head(10).to_html(
                index=False,
                border=0,
                classes="report-table",
                float_format=lambda x: f"{x:,.2f}"
            )
        else:
            opportunity_html_table = "<p>当前数据不足以生成市场机会评价。</p>"

        if forecast_result is not None:
            forecast_html_table = forecast_result.to_html(
                index=False,
                border=0,
                classes="report-table",
                float_format=lambda x: f"{x:,.2f}"
            )
        else:
            forecast_html_table = "<p>当前没有可用的预测结果。</p>"

        ai_html = (
            "<pre class='ai-report'>"
            + html.escape(ai_report)
            + "</pre>"
            if ai_report
            else "<p>尚未生成AI诊断报告。</p>"
        )

        report_html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
        <meta charset="UTF-8">
        <title>第85类商品出口市场分析报告</title>
        <style>
        body {{
            font-family: "Microsoft YaHei", "Noto Sans CJK SC",
                         Arial, sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 28px;
            color: #253044;
            line-height: 1.8;
            background: #ffffff;
        }}
        h1 {{
            text-align: center;
            margin-top: 50px;
            margin-bottom: 10px;
            color: #17365D;
        }}
        .subtitle {{
            text-align: center;
            color: #68778C;
            margin-bottom: 45px;
        }}
        h2 {{
            color: #17365D;
            border-bottom: 2px solid #DCE5EF;
            padding-bottom: 8px;
            margin-top: 42px;
        }}
        h3 {{
            color: #31577D;
            margin-top: 28px;
        }}
        .summary {{
            background: #F3F6FA;
            padding: 20px;
            border-radius: 8px;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 18px 0;
            font-size: 13px;
        }}
        .report-table th,
        .report-table td {{
            border: 1px solid #DCE3EA;
            padding: 9px;
            text-align: center;
        }}
        .report-table th {{
            background: #EAF0F6;
        }}
        .ai-report {{
            white-space: pre-wrap;
            overflow-wrap: break-word;
            font-family: inherit;
            line-height: 1.8;
            background: #F7F8FA;
            padding: 20px;
            border-radius: 8px;
        }}
        .chart {{
            margin: 25px 0;
        }}
        .note {{
            color: #667085;
            font-size: 12px;
        }}
        @media print {{
            body {{
                max-width: none;
                margin: 15mm;
                padding: 0;
            }}
            h2, h3 {{
                break-after: avoid;
            }}
            .chart {{
                break-inside: avoid;
            }}
        }}
        </style>
        </head>
        <body>

        <h1>第85类商品出口市场分析报告</h1>
        <p class="subtitle">
        数据期间：{first_month:%Y年%m月}—{latest_month:%Y年%m月}
        </p>

        <h2>一、数据概况</h2>
        <div class="summary">
        <p>累计出口额：{total_export:,.2f} 元</p>
        <p>月均出口额：{monthly_average:,.2f} 元</p>
        <p>数据覆盖：{len(monthly_export)} 个月</p>
        <p>最新数据月份：{latest_month:%Y年%m月}</p>
        <p>最近12个月同比增长：{growth_text}</p>
        <p>前五大市场份额：{top5_share:.2f}%</p>
        <p>HHI市场集中度：{hhi_total:.2f}</p>
        </div>

        <h2>二、出口趋势分析</h2>
        <p>{html.escape(report_summary)}</p>
        <div class="chart">{trend_html}</div>

        <h2>三、主要出口市场</h2>
        <div class="chart">{market_html}</div>
        {market_html_table}

        <h2>四、市场机会与风险评价</h2>
        {opportunity_html_table}

        <h2>五、出口趋势预测</h2>
        <p>预测模型：{html.escape(forecast_model_name)}</p>
        {forecast_html_table}

        <h2>六、AI智能市场诊断</h2>
        {ai_html}

        <h2>七、结论与说明</h2>
        <p>{html.escape(conclusion_text)}</p>

        <p class="note">
        说明：本报告依据用户上传的海关统计数据生成。
        市场机会综合得分为平台自定义相对评价指标；
        趋势预测仅依据历史数据和模型回测结果，
        不构成对未来市场表现的保证。
        </p>

        </body>
        </html>
        """

        st.divider()

        st.download_button(
            label="下载完整市场分析报告（HTML）",
            data=report_html.encode("utf-8"),
            file_name="第85类商品出口市场分析报告.html",
            mime="text/html",
            type="primary",
            key="download_full_report"
        )

        st.caption(
            "下载后使用浏览器打开报告。"
            "如需PDF，可在浏览器中选择“打印”→“另存为PDF”。"
        )
