import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from inst_list import inst_list

st.set_page_config(
    layout="wide",
    page_title="Analysis of Financial Instruments by Murat Koptur",
    page_icon="📈",
)
st.title("Analysis of Financial Instruments")
col0, col1 = st.columns(2)
with col0:
    st.write("Analyze financial instruments' past performances with a just few clicks.")
    st.write(
        f"Contact me: [LinkedIn](https://www.linkedin.com/in/muratkoptur/) [GitHub](https://github.com/mrtkp9993) [Email](mailto:contact@muratkoptur.com)"
    )
    st.write(
        f"Support me: [GitHub Sponsor](https://github.com/sponsors/mrtkp9993) [Patreon](https://www.patreon.com/muratkoptur)"
    )
    st.write("Made with ❤️ by [Murat Koptur](https://muratkoptur.com)")

with col1:
    st.write(
        """
    Check my Android apps

    [tradeslyFX Forex AI Roboadvisor](https://play.google.com/store/apps/details?id=com.tradesly.tradeslyfx)

    [tradeslyPro Cryptocurrency AI Roboadvisor](https://play.google.com/store/apps/details?id=com.tradesly.tradeslypro)
    """
    )
st.divider()

benchmark_instruments = {
    "^IRX": "13 Week Treasury Bill",
    "^FVX": "Treasury Yield 5 Years",
    "^TNX": "Treasury Yield 10 Years",
    "^TYX": "Treasury Yield 30 Years",
}

symbols = inst_list


@st.cache_data
def download_benchmark_data():
    benchmark_data = yf.download(list(benchmark_instruments.keys()), period="2y")
    benchmark_data = benchmark_data["Close"]
    benchmark_data = benchmark_data.fillna(method="ffill")
    benchmark_data = benchmark_data.dropna()
    return benchmark_data


with st.spinner("Downloading Benchmark Data..."):
    benchmark_data = download_benchmark_data()

col2, col3 = st.columns(2)
with col2:
    selected_ins = st.selectbox(
        "Select a financial instrument",
        symbols,
        format_func=lambda x: f"{x} - {symbols[x]}",
    )
with col3:
    selected_bench = st.selectbox(
        "Select a benchmark instrument",
        benchmark_instruments,
        index=1,
        format_func=lambda x: f"{x} - {benchmark_instruments[x]}",
    )


error = 0
if st.button("Run Analysis"):
    with st.spinner("Analyzing..."):
        try:
            df = yf.download(selected_ins, period="2y")
            df = df["Close"]
            join_df = pd.concat([df, benchmark_data[selected_bench]], axis=1)
            join_df.columns = [selected_ins, selected_bench]
            join_df = join_df.dropna()
            join_df_change = join_df.pct_change()
            excess_returns = (
                join_df_change[selected_ins] - join_df_change[selected_bench]
            )
            excess_returns.name = "Excess Returns"

            stat_table = pd.DataFrame(
                columns=["Benchmark", "Instrument", "Excess Returns"]
            )
            stat_table.loc["Mean"] = [
                join_df_change[selected_bench].mean(),
                join_df_change[selected_ins].mean(),
                excess_returns.mean(),
            ]
            stat_table.loc["Standart Deviation"] = [
                join_df_change[selected_bench].std(),
                join_df_change[selected_ins].std(),
                excess_returns.std(),
            ]
            stat_table.loc["Information Ratio"] = [
                np.nan,
                np.nan,
                excess_returns.mean() / excess_returns.std(),
            ]
            stat_table.loc["Value at Risk"] = [
                join_df_change[selected_bench].quantile(0.05),
                join_df_change[selected_ins].quantile(0.05),
                excess_returns.quantile(0.05),
            ]
            stat_table.loc["Calmar Ratio"] = [
                np.nan,
                np.nan,
                excess_returns.mean() / excess_returns.abs().rolling(252).mean().min(),
            ]
            stat_table.loc["Skewness"] = [
                join_df_change[selected_bench].skew(),
                join_df_change[selected_ins].skew(),
                excess_returns.skew(),
            ]
            stat_table.loc["Kurtosis"] = [
                join_df_change[selected_bench].kurtosis(),
                join_df_change[selected_ins].kurtosis(),
                excess_returns.kurtosis(),
            ]
            stat_table.loc["Appraisal Ratio"] = [
                np.nan,
                np.nan,
                excess_returns.mean() / excess_returns.abs().mean(),
            ]
            stat_table.loc["Batting Average"] = [
                (join_df_change[selected_bench] > 0).sum()
                / join_df_change[selected_bench].count(),
                (join_df_change[selected_ins] > 0).sum()
                / join_df_change[selected_ins].count(),
                (excess_returns > 0).sum() / excess_returns.count(),
            ]
            stat_table.loc["Upside Potential Ratio"] = [
                join_df_change[selected_bench][
                    join_df_change[selected_bench] > 0
                ].mean()
                / join_df_change[selected_bench][
                    join_df_change[selected_bench] < 0
                ].mean(),
                join_df_change[selected_ins][join_df_change[selected_ins] > 0].mean()
                / join_df_change[selected_ins][join_df_change[selected_ins] < 0].mean(),
                excess_returns[excess_returns > 0].mean()
                / excess_returns[excess_returns < 0].mean(),
            ]
            stat_table.loc["Gain-Loss Ratio"] = [
                join_df_change[selected_bench][join_df_change[selected_bench] > 0].sum()
                / join_df_change[selected_bench][
                    join_df_change[selected_bench] < 0
                ].sum(),
                join_df_change[selected_ins][join_df_change[selected_ins] > 0].sum()
                / join_df_change[selected_ins][join_df_change[selected_ins] < 0].sum(),
                excess_returns[excess_returns > 0].sum()
                / excess_returns[excess_returns < 0].sum(),
            ]
            error = 0
        except:
            error = 1

    if error:
        st.error("Failed to analyze.")
    else:
        st.success("Analysis completed.")
        with st.expander("Results"):
            col4, col5 = st.columns(2)
            with col4:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=join_df.index, y=join_df[selected_ins], name=f"{selected_ins}"
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=join_df.index,
                        y=join_df[selected_bench],
                        name=f"{selected_bench}",
                    ),
                    secondary_y=True,
                )
                fig.update_layout(
                    title_text="Price History (2 Years)",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col5:
                fig = make_subplots()
                fig.add_trace(
                    go.Scatter(
                        x=excess_returns.index,
                        y=excess_returns,
                        name="Excess Returns",
                    )
                )
                fig.update_layout(
                    title_text="Excess Returns (2 Years)",
                )
                st.plotly_chart(fig, use_container_width=True)
            col6, col7 = st.columns(2)
            with col6:
                fig = px.scatter(
                    data_frame=join_df_change,
                    x=selected_bench,
                    y=selected_ins,
                    trendline="ols",
                    trendline_color_override="purple",
                )
                st.plotly_chart(fig, use_container_width=True)
                model = px.get_trendline_results(fig)
                beta = model.iloc[0]["px_fit_results"].params[1]
                alpha = model.iloc[0]["px_fit_results"].params[0]
                stat_table.loc["Beta"] = [np.nan, beta, np.nan]
                stat_table.loc["Alpha"] = [np.nan, alpha, np.nan]
                stat_table.sort_index(inplace=True)
            with col7:
                st.write("Return Statistics")
                st.dataframe(stat_table, use_container_width=True)
