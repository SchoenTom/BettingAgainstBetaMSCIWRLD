"""
dashboard.py - Interactive Streamlit dashboard for BAB strategy analysis

This script provides an interactive web dashboard to explore:
- Cumulative equity curves (BAB vs MSCI World)
- Rolling performance metrics
- Quintile statistics
- Drawdown analysis

Usage:
    streamlit run dashboard.py

Note: This dashboard reads only from local CSV files - no API calls.

Author: BAB Strategy Implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="BAB Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@st.cache_data
def load_data():
    """
    Load all required data from CSV files.

    Returns:
        tuple: (monthly_perf, bab_portfolio, quintile_returns, summary)
    """
    # Load monthly performance
    perf_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    monthly_perf = pd.read_csv(perf_path, index_col=0, parse_dates=True)

    # Load BAB portfolio
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    bab_portfolio = pd.read_csv(bab_path, index_col=0, parse_dates=True)

    # Load quintile returns
    quintile_path = os.path.join(OUTPUT_DIR, 'quintile_returns.csv')
    quintile_returns = pd.read_csv(quintile_path, index_col=0, parse_dates=True)

    # Load summary
    summary_path = os.path.join(OUTPUT_DIR, 'bab_backtest_summary.csv')
    summary = pd.read_csv(summary_path)

    return monthly_perf, bab_portfolio, quintile_returns, summary


def check_data_exists():
    """
    Check if required data files exist.

    Returns:
        bool: True if all files exist
    """
    required_files = [
        os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv'),
        os.path.join(OUTPUT_DIR, 'bab_portfolio.csv'),
        os.path.join(OUTPUT_DIR, 'quintile_returns.csv'),
        os.path.join(OUTPUT_DIR, 'bab_backtest_summary.csv')
    ]
    return all(os.path.exists(f) for f in required_files)


def format_percent(value, decimals=2):
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value, decimals=3):
    """Format number with decimals."""
    return f"{value:.{decimals}f}"


def create_equity_curve_plot(monthly_perf, log_scale=True):
    """
    Create interactive cumulative returns chart.

    Args:
        monthly_perf: Performance DataFrame
        log_scale: Whether to use log scale

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_perf.index,
        y=monthly_perf['BAB_Cumulative'],
        mode='lines',
        name='BAB Strategy',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=monthly_perf.index,
        y=monthly_perf['Benchmark_Cumulative'],
        mode='lines',
        name='MSCI World',
        line=dict(color='#ff7f0e', width=2)
    ))

    # Add reference line at $1
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Cumulative Returns: BAB Strategy vs MSCI World',
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        yaxis_type='log' if log_scale else 'linear',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )

    return fig


def create_rolling_sharpe_plot(monthly_perf):
    """
    Create rolling Sharpe ratio chart.

    Args:
        monthly_perf: Performance DataFrame

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_perf.index,
        y=monthly_perf['Rolling_12M_BAB_Sharpe'],
        mode='lines',
        name='BAB Strategy',
        line=dict(color='#1f77b4', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=monthly_perf.index,
        y=monthly_perf['Rolling_12M_Benchmark_Sharpe'],
        mode='lines',
        name='MSCI World',
        line=dict(color='#ff7f0e', width=1.5)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)

    fig.update_layout(
        title='Rolling 12-Month Sharpe Ratio',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400
    )

    return fig


def create_drawdown_plot(monthly_perf):
    """
    Create drawdown chart.

    Args:
        monthly_perf: Performance DataFrame

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_perf.index,
        y=monthly_perf['BAB_Drawdown'] * 100,
        fill='tozeroy',
        name='BAB Strategy',
        line=dict(color='#1f77b4'),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.add_trace(go.Scatter(
        x=monthly_perf.index,
        y=monthly_perf['Benchmark_Drawdown'] * 100,
        fill='tozeroy',
        name='MSCI World',
        line=dict(color='#ff7f0e'),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))

    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        height=400
    )

    return fig


def create_beta_spread_plot(bab_portfolio):
    """
    Create beta spread chart.

    Args:
        bab_portfolio: BAB portfolio DataFrame

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bab_portfolio.index,
        y=bab_portfolio['Beta_Spread'],
        mode='lines',
        name='Beta Spread',
        line=dict(color='#9467bd', width=1.5)
    ))

    # Add rolling mean
    rolling_spread = bab_portfolio['Beta_Spread'].rolling(12).mean()
    fig.add_trace(go.Scatter(
        x=bab_portfolio.index,
        y=rolling_spread,
        mode='lines',
        name='12-Month Rolling Mean',
        line=dict(color='darkred', width=2, dash='dash')
    ))

    avg_spread = bab_portfolio['Beta_Spread'].mean()
    fig.add_hline(y=avg_spread, line_dash="dot", line_color="gray",
                  annotation_text=f"Avg: {avg_spread:.2f}")

    fig.update_layout(
        title='Beta Spread (Q5 High Beta - Q1 Low Beta)',
        xaxis_title='Date',
        yaxis_title='Beta Spread',
        hovermode='x unified',
        height=400
    )

    return fig


def create_quintile_returns_plot(quintile_returns):
    """
    Create quintile returns bar chart.

    Args:
        quintile_returns: Quintile returns DataFrame

    Returns:
        Plotly figure
    """
    return_cols = [f'Q{i}_Return' for i in range(1, 6)]
    avg_returns = quintile_returns[return_cols].mean() * 100

    colors = ['#2ca02c', '#5aa02c', '#b8860b', '#d63a28', '#d62728']

    fig = go.Figure(data=[
        go.Bar(
            x=['Q1<br>(Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5<br>(High Beta)'],
            y=avg_returns.values,
            marker_color=colors,
            text=[f'{v:.2f}%' for v in avg_returns.values],
            textposition='outside'
        )
    ])

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title='Average Monthly Returns by Beta Quintile',
        xaxis_title='Beta Quintile',
        yaxis_title='Average Monthly Return (%)',
        height=400,
        showlegend=False
    )

    return fig


def create_quintile_betas_plot(quintile_returns):
    """
    Create quintile average betas bar chart.

    Args:
        quintile_returns: Quintile returns DataFrame

    Returns:
        Plotly figure
    """
    beta_cols = [f'Q{i}_Mean_Beta' for i in range(1, 6)]
    avg_betas = quintile_returns[beta_cols].mean()

    colors = ['#2ca02c', '#5aa02c', '#b8860b', '#d63a28', '#d62728']

    fig = go.Figure(data=[
        go.Bar(
            x=['Q1<br>(Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5<br>(High Beta)'],
            y=avg_betas.values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in avg_betas.values],
            textposition='outside'
        )
    ])

    fig.add_hline(y=1, line_dash="dash", line_color="gray",
                  annotation_text="Market Beta = 1")

    fig.update_layout(
        title='Average Beta by Quintile',
        xaxis_title='Beta Quintile',
        yaxis_title='Average Beta',
        height=400,
        showlegend=False
    )

    return fig


def create_yearly_returns_plot(monthly_perf):
    """
    Create yearly returns comparison chart.

    Args:
        monthly_perf: Performance DataFrame

    Returns:
        Plotly figure
    """
    yearly_bab = (1 + monthly_perf['BAB_Return']).resample('YE').prod() - 1
    yearly_bench = (1 + monthly_perf['Benchmark_Return']).resample('YE').prod() - 1

    years = [str(y) for y in yearly_bab.index.year]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='BAB Strategy',
        x=years,
        y=yearly_bab.values * 100,
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        name='MSCI World',
        x=years,
        y=yearly_bench.values * 100,
        marker_color='#ff7f0e'
    ))

    fig.add_hline(y=0, line_color="black", line_width=0.5)

    fig.update_layout(
        title='Annual Returns Comparison',
        xaxis_title='Year',
        yaxis_title='Annual Return (%)',
        barmode='group',
        height=400
    )

    return fig


def display_metrics_cards(summary):
    """
    Display key metrics in card format.

    Args:
        summary: Summary DataFrame
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="BAB Annualized Return",
            value=format_percent(summary['Annualized_Return'].iloc[0]),
            delta=format_percent(summary['Annualized_Return'].iloc[0] -
                                summary['Benchmark_Ann_Return'].iloc[0]) + " vs MSCI"
        )

    with col2:
        st.metric(
            label="BAB Sharpe Ratio",
            value=format_number(summary['Sharpe_Ratio'].iloc[0]),
            delta=format_number(summary['Sharpe_Ratio'].iloc[0] -
                               summary['Benchmark_Sharpe'].iloc[0]) + " vs MSCI"
        )

    with col3:
        st.metric(
            label="BAB Max Drawdown",
            value=format_percent(summary['Max_Drawdown'].iloc[0]),
            delta=format_percent(summary['Max_Drawdown'].iloc[0] -
                                summary['Benchmark_Max_DD'].iloc[0]) + " vs MSCI",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="BAB Win Rate",
            value=format_percent(summary['Win_Rate'].iloc[0]),
        )


def display_detailed_metrics(summary):
    """
    Display detailed performance metrics.

    Args:
        summary: Summary DataFrame
    """
    s = summary.iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BAB Strategy Metrics")
        metrics_bab = {
            "Total Return": format_percent(s['Total_Return']),
            "Annualized Return": format_percent(s['Annualized_Return']),
            "Annualized Volatility": format_percent(s['Annualized_Volatility']),
            "Sharpe Ratio": format_number(s['Sharpe_Ratio']),
            "Sortino Ratio": format_number(s['Sortino_Ratio']),
            "Calmar Ratio": format_number(s['Calmar_Ratio']),
            "Max Drawdown": format_percent(s['Max_Drawdown']),
            "Win Rate": format_percent(s['Win_Rate']),
            "Best Month": format_percent(s['Best_Month']),
            "Worst Month": format_percent(s['Worst_Month']),
            "Skewness": format_number(s['Skewness']),
            "Kurtosis": format_number(s['Kurtosis']),
        }
        st.dataframe(pd.DataFrame.from_dict(metrics_bab, orient='index',
                                            columns=['Value']),
                     use_container_width=True)

    with col2:
        st.subheader("Risk-Adjusted & Relative Metrics")
        metrics_rel = {
            "Beta to Benchmark": format_number(s['Beta_to_Benchmark']),
            "Alpha (Annualized)": format_percent(s['Alpha']),
            "Information Ratio": format_number(s['Information_Ratio']),
            "---": "---",
            "Benchmark Ann Return": format_percent(s['Benchmark_Ann_Return']),
            "Benchmark Ann Vol": format_percent(s['Benchmark_Ann_Vol']),
            "Benchmark Sharpe": format_number(s['Benchmark_Sharpe']),
            "Benchmark Max DD": format_percent(s['Benchmark_Max_DD']),
        }
        st.dataframe(pd.DataFrame.from_dict(metrics_rel, orient='index',
                                            columns=['Value']),
                     use_container_width=True)


def display_quintile_stats(bab_portfolio, quintile_returns):
    """
    Display quintile statistics.

    Args:
        bab_portfolio: BAB portfolio DataFrame
        quintile_returns: Quintile returns DataFrame
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quintile Summary")

        # Average statistics
        avg_stats = {
            'Quintile': ['Q1 (Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5 (High Beta)'],
            'Avg Beta': [quintile_returns[f'Q{i}_Mean_Beta'].mean() for i in range(1, 6)],
            'Avg Monthly Return (%)': [quintile_returns[f'Q{i}_Return'].mean() * 100 for i in range(1, 6)],
            'Avg # Stocks': [quintile_returns[f'Q{i}_N'].mean() for i in range(1, 6)]
        }
        stats_df = pd.DataFrame(avg_stats)
        stats_df['Avg Beta'] = stats_df['Avg Beta'].map('{:.3f}'.format)
        stats_df['Avg Monthly Return (%)'] = stats_df['Avg Monthly Return (%)'].map('{:.3f}'.format)
        stats_df['Avg # Stocks'] = stats_df['Avg # Stocks'].map('{:.1f}'.format)

        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("BAB Strategy Summary")

        bab_stats = {
            'Metric': ['Avg Beta Spread', 'Avg Q1 Beta', 'Avg Q5 Beta',
                      'Avg Q1 Stocks', 'Avg Q5 Stocks', 'Avg Total Stocks'],
            'Value': [
                f"{bab_portfolio['Beta_Spread'].mean():.3f}",
                f"{bab_portfolio['Q1_Mean_Beta'].mean():.3f}",
                f"{bab_portfolio['Q5_Mean_Beta'].mean():.3f}",
                f"{bab_portfolio['N_Q1'].mean():.1f}",
                f"{bab_portfolio['N_Q5'].mean():.1f}",
                f"{bab_portfolio['N_Total'].mean():.1f}"
            ]
        }

        st.dataframe(pd.DataFrame(bab_stats), use_container_width=True, hide_index=True)


def main():
    """
    Main dashboard application.
    """
    st.title("📈 Betting-Against-Beta (BAB) Strategy Dashboard")
    st.markdown("---")

    # Check if data exists
    if not check_data_exists():
        st.error("""
        ⚠️ **Data files not found!**

        Please run the following scripts first to generate the required data:

        ```bash
        python data_loader.py
        python portfolio_construction.py
        python backtest.py
        ```

        After running these scripts, refresh this dashboard.
        """)
        return

    # Load data
    try:
        monthly_perf, bab_portfolio, quintile_returns, summary = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar
    st.sidebar.header("Dashboard Controls")

    # Date range filter
    st.sidebar.subheader("Date Range")
    min_date = monthly_perf.index.min().to_pydatetime()
    max_date = monthly_perf.index.max().to_pydatetime()

    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Filter data by date range
    mask = (monthly_perf.index >= pd.Timestamp(start_date)) & (monthly_perf.index <= pd.Timestamp(end_date))
    filtered_perf = monthly_perf.loc[mask].copy()
    filtered_bab = bab_portfolio.loc[bab_portfolio.index.isin(filtered_perf.index)].copy()
    filtered_quintile = quintile_returns.loc[quintile_returns.index.isin(filtered_perf.index)].copy()

    # Recalculate cumulative returns for filtered period
    filtered_perf['BAB_Cumulative'] = (1 + filtered_perf['BAB_Return']).cumprod()
    filtered_perf['Benchmark_Cumulative'] = (1 + filtered_perf['Benchmark_Return']).cumprod()

    # Log scale toggle
    log_scale = st.sidebar.checkbox("Log Scale (Equity Curve)", value=True)

    # Display options
    st.sidebar.subheader("Display Options")
    show_metrics = st.sidebar.checkbox("Show Key Metrics", value=True)
    show_equity = st.sidebar.checkbox("Show Equity Curve", value=True)
    show_rolling = st.sidebar.checkbox("Show Rolling Metrics", value=True)
    show_drawdown = st.sidebar.checkbox("Show Drawdowns", value=True)
    show_quintiles = st.sidebar.checkbox("Show Quintile Analysis", value=True)
    show_yearly = st.sidebar.checkbox("Show Yearly Returns", value=True)

    # Info
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Data Period:**
    {summary['Start_Date'].iloc[0]} to {summary['End_Date'].iloc[0]}

    **Total Months:** {summary['N_Months'].iloc[0]}
    """)

    # Main content
    if show_metrics:
        st.header("📊 Key Performance Metrics")
        display_metrics_cards(summary)

        with st.expander("View Detailed Metrics"):
            display_detailed_metrics(summary)

    if show_equity:
        st.header("📈 Cumulative Returns")
        st.plotly_chart(create_equity_curve_plot(filtered_perf, log_scale),
                       use_container_width=True)

    if show_rolling:
        st.header("📉 Rolling Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_rolling_sharpe_plot(filtered_perf),
                           use_container_width=True)
        with col2:
            st.plotly_chart(create_beta_spread_plot(filtered_bab),
                           use_container_width=True)

    if show_drawdown:
        st.header("⚠️ Drawdown Analysis")
        st.plotly_chart(create_drawdown_plot(filtered_perf),
                       use_container_width=True)

    if show_quintiles:
        st.header("📊 Quintile Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_quintile_returns_plot(filtered_quintile),
                           use_container_width=True)
        with col2:
            st.plotly_chart(create_quintile_betas_plot(filtered_quintile),
                           use_container_width=True)

        display_quintile_stats(filtered_bab, filtered_quintile)

    if show_yearly:
        st.header("📅 Annual Returns")
        st.plotly_chart(create_yearly_returns_plot(filtered_perf),
                       use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>BAB Strategy Dashboard | Data sourced from yfinance | Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
