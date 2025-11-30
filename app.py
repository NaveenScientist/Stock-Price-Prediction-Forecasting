import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ast
import yfinance as yf
from datetime import datetime, timedelta
import time

# ----------------- CONFIGURATION -----------------
st.set_page_config(
    page_title="Nifty50 Pro Analytics & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CONSTANTS -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = BASE_DIR
RESULTS_FOLDER = os.path.join(BASE_DIR, "Results_MultiFeature_7day")
SUMMARY_FILE = os.path.join(RESULTS_FOLDER, "Nifty50_7day_summary.csv")

# Ensure directories exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Symbol Mapping (Stock Name -> Yahoo Ticker)
SYMBOL_MAP = {
    'Adani Ent': 'ADANIENT.NS', 'Adani Ports': 'ADANIPORTS.NS', 'Airtel': 'BHARTIARTL.NS',
    'Apollo Hosp': 'APOLLOHOSP.NS', 'Asian paints': 'ASIANPAINT.NS', 'Axis bank': 'AXISBANK.NS',
    'Bajaj auto': 'BAJAJ-AUTO.NS', 'Bajaj finance': 'BAJFINANCE.NS', 'bajaj FinServ': 'BAJAJFINSV.NS',
    'Bharat electronics': 'BEL.NS', 'Britannia': 'BRITANNIA.NS', 'Cipla': 'CIPLA.NS',
    'Coal india': 'COALINDIA.NS', 'Divis Lab': 'DIVISLAB.NS', 'Dr. Reddy': 'DRREDDY.NS',
    'Eicher motors': 'EICHERMOT.NS', 'Grasim': 'GRASIM.NS', 'HCL tech': 'HCLTECH.NS',
    'HDFC bnk': 'HDFCBANK.NS', 'HDFC Life': 'HDFCLIFE.NS', 'Hero moto': 'HEROMOTOCO.NS',
    'Hindalco': 'HINDALCO.NS', 'HUL': 'HINDUNILVR.NS', 'ICICI bnk': 'ICICIBANK.NS',
    'Indusind bnk': 'INDUSINDBK.NS', 'Infosys': 'INFY.NS', 'ITC': 'ITC.NS',
    'JIO finance': 'JIOFIN.NS', 'JSW steel': 'JSWSTEEL.NS', 'Kotak mahindra bnk': 'KOTAKBANK.NS',
    'L&T': 'LT.NS', 'M&M': 'M&M.NS', 'Maruti Suzuki': 'MARUTI.NS', 'Nestle ind': 'NESTLEIND.NS',
    'NTPC': 'NTPC.NS', 'ONGC': 'ONGC.NS', 'Power Grid': 'POWERGRID.NS', 'Reliance': 'RELIANCE.NS',
    'SBI': 'SBIN.NS', 'SBI life': 'SBILIFE.NS', 'Shriram finance': 'SHRIRAMFIN.NS',
    'SUN pharma': 'SUNPHARMA.NS', 'TATA consumer': 'TATACONSUM.NS', 'TATA motors': 'TATAMOTORS.NS',
    'TATA steel': 'TATASTEEL.NS', 'TCS': 'TCS.NS', 'Tech M': 'TECHM.NS', 'Titan': 'TITAN.NS',
    'Trent': 'TRENT.NS', 'Ultra Tech cem': 'ULTRACEMCO.NS', 'Wipro': 'WIPRO.NS'
}

# ----------------- STYLING -----------------
st.markdown("""
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #1f242d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    h1 {
        background: linear-gradient(90deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
        font-family: 'Source Code Pro', monospace;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #2ea043;
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot .plotly .modebar {
        orientation: v;
        top: 0;
        right: -30px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA UPDATE FUNCTIONS -----------------
def update_stock_data():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    updated_count = 0
    total_files = len([f for f in os.listdir(DATA_FOLDER) if f.startswith("N50 Stock data  - ") and f.endswith(".csv")])
    
    processed = 0
    
    for filename in os.listdir(DATA_FOLDER):
        if filename.startswith("N50 Stock data  - ") and filename.endswith(".csv"):
            stock_name = filename.replace("N50 Stock data  - ", "").replace(".csv", "")
            filepath = os.path.join(DATA_FOLDER, filename)
            
            ticker_symbol = SYMBOL_MAP.get(stock_name)
            if not ticker_symbol:
                processed += 1
                continue
                
            try:
                # Load existing data
                df = pd.read_csv(filepath)
                
                # Parse Dates
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                
                # Get last date
                if not df['Date'].empty:
                    last_date = df['Date'].max()
                else:
                    last_date = datetime.now() - timedelta(days=365)
                
                # Check if update is needed
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                if last_date >= today:
                    processed += 1
                    progress_bar.progress(processed / total_files)
                    continue
                
                status_text.text(f"Updating {stock_name}...")
                
                # Fetch new data
                start_date = last_date + timedelta(days=1)
                ticker = yf.Ticker(ticker_symbol)
                new_data = ticker.history(start=start_date, interval="1d")
                
                if not new_data.empty:
                    new_rows = []
                    for idx, row in new_data.iterrows():
                        date_str = idx.strftime('%d/%m/%Y 15:30:00')
                        new_rows.append({
                            'Date': date_str,
                            'Open': round(row['Open'], 2),
                            'High': round(row['High'], 2),
                            'Low': round(row['Low'], 2),
                            'Close': round(row['Close'], 2),
                            'Volume': int(row['Volume'])
                        })
                    
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        df['Date'] = df['Date'].dt.strftime('%d/%m/%Y %H:%M:%S')
                        final_df = pd.concat([df, new_df], ignore_index=True)
                        final_df.to_csv(filepath, index=False)
                        updated_count += 1
                        
            except Exception as e:
                print(f"Error updating {stock_name}: {e}")
            
            processed += 1
            progress_bar.progress(processed / total_files)
            
    status_text.text(f"Update Complete! Updated {updated_count} files.")
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()
    return updated_count

# ----------------- HELPER FUNCTIONS -----------------
def get_live_price(stock_name):
    symbol = SYMBOL_MAP.get(stock_name)
    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
    return None

@st.cache_data(show_spinner=False)
def load_summary(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    
    # Load Sector Mapping
    mapping_file = os.path.join(DATA_FOLDER, "N50_sector_mapping.csv")
    if os.path.exists(mapping_file):
        sector_map_df = pd.read_csv(mapping_file)
        
        manual_map = {
             'Adani Ent': 'Adani Enterprises Ltd.', 'Adani Ports': 'Adani Ports and SEZ Ltd.', 'Airtel': 'Bharti Airtel Ltd.',
             'Apollo Hosp': 'Apollo Hospitals Enterprise Ltd.', 'Asian paints': 'Asian Paints Ltd.', 'Axis bank': 'Axis Bank Ltd.',
             'Bajaj auto': 'Bajaj Auto Ltd.', 'Bajaj finance': 'Bajaj Finance Ltd.', 'bajaj FinServ': 'Bajaj Finserv Ltd.',
             'Bharat electronics': 'Bharat Electronics Ltd.', 'Britannia': 'Britannia Industries Ltd.', 'Cipla': 'Cipla Ltd.',
             'Coal india': 'Coal India Ltd.', 'Divis Lab': "Divi's Laboratories Ltd.", 'Dr. Reddy': "Dr. Reddy's Laboratories Ltd.",
             'Eicher motors': 'Eicher Motors Ltd.', 'Grasim': 'Grasim Industries Ltd.', 'HCL tech': 'HCL Technologies Ltd.',
             'HDFC bnk': 'HDFC Bank Ltd.', 'HDFC Life': 'HDFC Life Insurance Company Ltd.', 'Hero moto': 'Hero MotoCorp Ltd.',
             'Hindalco': 'Hindalco Industries Ltd.', 'HUL': 'Hindustan Unilever Ltd.', 'ICICI bnk': 'ICICI Bank Ltd.',
             'Indusind bnk': 'IndusInd Bank Ltd.', 'Infosys': 'Infosys Ltd.', 'ITC': 'ITC Ltd.',
             'JIO finance': 'Jio Financial Services Ltd.', 'JSW steel': 'JSW Steel Ltd.', 'Kotak mahindra bnk': 'Kotak Mahindra Bank Ltd.',
             'L&T': 'Larsen & Toubro Ltd.', 'M&M': 'Mahindra & Mahindra Ltd.', 'Maruti Suzuki': 'Maruti Suzuki India Ltd.',
             'Nestle ind': 'Nestle India Ltd.', 'NTPC': 'NTPC Ltd.', 'ONGC': 'Oil and Natural Gas Corporation Ltd.',
             'Power Grid': 'Power Grid Corporation of India Ltd.', 'Reliance': 'Reliance Industries Ltd.', 'SBI': 'State Bank of India',
             'SBI life': 'SBI Life Insurance Company Ltd.', 'Shriram finance': 'Shriram Finance Ltd.', 'SUN pharma': 'Sun Pharmaceutical Industries Ltd.',
             'TATA consumer': 'Tata Consumer Products Ltd.', 'TATA motors': 'Tata Motors Ltd.', 'TATA steel': 'Tata Steel Ltd.',
             'TCS': 'Tata Consultancy Services Ltd.', 'Tech M': 'Tech Mahindra Ltd.', 'Titan': 'Titan Company Ltd.',
             'Trent': 'Trent Ltd.', 'Ultra Tech cem': 'UltraTech Cement Ltd.', 'Wipro': 'Wipro Ltd.'
        }
        
        def get_sector(stock_name):
            full_name = manual_map.get(stock_name)
            if full_name:
                match = sector_map_df[sector_map_df['Stock'] == full_name]
                if not match.empty:
                    return match.iloc[0]['Sector']
            for _, row in sector_map_df.iterrows():
                if stock_name.lower() in row['Stock'].lower():
                    return row['Sector']
            return "Unknown"

        df['Sector'] = df['Stock'].apply(get_sector)
    
    def parse_next7(x):
        if pd.isna(x): return []
        if isinstance(x, list): return x
        try:
            return ast.literal_eval(x)
        except:
            try:
                return [float(i) for i in str(x).split(",")]
            except:
                return []
                
    df['Next_7_List'] = df['Next_7_Days'].apply(parse_next7)
    return df

@st.cache_data(show_spinner=False)
def load_stock_history(stock_name):
    filename = f"N50 Stock data  - {stock_name}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    
    if not os.path.exists(filepath):
        for f in os.listdir(DATA_FOLDER):
            if stock_name.lower() in f.lower() and f.endswith('.csv'):
                filepath = os.path.join(DATA_FOLDER, f)
                break
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            # If format failed, try standard
            if df['Date'].isna().all():
                 df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        return df
    return None

# ----------------- MAIN APP -----------------
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("📊 Nifty50 Analytics")
        st.markdown("---")
        
        # Update Button
        if st.button("🔄 Update All Data", help="Fetch latest prices from Yahoo Finance"):
            update_stock_data()
            st.cache_data.clear() # Clear cache to reload new data
            st.success("Data updated successfully!")
            
        st.markdown("---")
        
        nav_mode = st.radio("Navigation", ["Dashboard", "Stock Analysis", "Market Overview"])
        
        st.markdown("---")
        st.subheader("Filters")
        
        summary = load_summary(SUMMARY_FILE)
        if summary is None:
             st.error("Summary file missing. Please run analysis first.")
             st.stop()

        sectors = ['All'] + sorted(summary['Sector'].dropna().unique().tolist())
        sel_sector = st.selectbox("Select Sector", sectors)
        
        if sel_sector == 'All':
            filtered_df = summary.copy()
        else:
            filtered_df = summary[summary['Sector'] == sel_sector]
            
        stock_list = sorted(filtered_df['Stock'].tolist())
        sel_stock = st.selectbox("Select Stock", stock_list)

    # --- PAGES ---
    if nav_mode == "Dashboard":
        render_dashboard(summary)
        
    elif nav_mode == "Stock Analysis":
        render_stock_analysis(sel_stock, summary)
        
    elif nav_mode == "Market Overview":
        render_market_overview(summary)

# ----------------- PAGE FUNCTIONS -----------------
def render_dashboard(summary):
    st.title("🚀 Market Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(summary))
    with col2:
        st.metric("Avg R² Score", f"{summary['R2_day1'].mean():.4f}")
    with col3:
        best_stock = summary.loc[summary['R2_day1'].idxmax()]
        st.metric("Best Model", best_stock['Stock'], delta=f"R² {best_stock['R2_day1']:.3f}")
    with col4:
        worst_stock = summary.loc[summary['R2_day1'].idxmin()]
        st.metric("Needs Improvement", worst_stock['Stock'], delta=f"R² {worst_stock['R2_day1']:.3f}", delta_color="inverse")
        
    st.markdown("---")
    
    st.subheader("Sector Performance (Average R² Score)")
    sector_perf = summary.groupby('Sector')['R2_day1'].mean().reset_index().sort_values('R2_day1', ascending=False)
    
    fig = px.bar(sector_perf, x='Sector', y='R2_day1', color='R2_day1', color_continuous_scale='Viridis', template="plotly_dark")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def render_stock_analysis(stock_name, summary):
    st.title(f"📈 {stock_name} Analysis")
    
    stock_data = summary[summary['Stock'] == stock_name].iloc[0]
    
    next7 = stock_data['Next_7_List']
    source = "Cached Summary"

    # Metrics
    live_price = get_live_price(stock_name)
    next_day_pred = next7[0] if len(next7) > 0 else 0
    r2 = stock_data['R2_day1']
    
    st.caption(f"Data Source: {source}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector", stock_data['Sector'])
    col2.metric("Model Accuracy (R²)", f"{r2:.4f}")
    
    if live_price:
        delta = live_price - next_day_pred
        col3.metric("Live Price", f"₹{live_price:.2f}")
        col4.metric("Next Day Pred", f"₹{next_day_pred:,.2f}", delta=f"{delta:.2f} vs Live")
    else:
        col4.metric("Next Day Pred", f"₹{next_day_pred:,.2f}")

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Price Forecast & History", "Forecast Data"])
    
    with tab1:
        history_df = load_stock_history(stock_name)
        
        if history_df is not None:
            fig = go.Figure()
            
            # Historical Data
            display_hist = history_df.tail(180)
            
            fig.add_trace(go.Candlestick(
                x=display_hist['Date'],
                open=display_hist['Open'],
                high=display_hist['High'],
                low=display_hist['Low'],
                close=display_hist['Close'],
                name='Historical OHLC'
            ))
            
            # Forecast Data
            if next7:
                last_date = display_hist['Date'].iloc[-1]
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=next7,
                    mode='lines+markers',
                    name='7-Day Forecast',
                    line=dict(color='#00ff88', width=3, dash='dot'),
                    marker=dict(size=8, color='#00ff88')
                ))
            
            fig.update_layout(
                title=f"{stock_name} Price History & Forecast",
                yaxis_title="Price (INR)",
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Historical data file not found.")

    with tab2:
        st.subheader("Detailed Forecast Values")
        if next7:
            df_next = pd.DataFrame({
                'Day': [f"Day +{i}" for i in range(1, 8)],
                'Predicted Price': next7
            })
            st.table(df_next.style.format({"Predicted Price": "₹{:,.2f}"}))

def render_market_overview(summary):
    st.title("🌐 Market Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("R² vs RMSE Distribution")
        fig = px.scatter(
            summary,
            x='R2_day1',
            y='RMSE_day1',
            color='Sector',
            hover_data=['Stock'],
            title="Model Accuracy Landscape",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Sector Breakdown")
        sector_counts = summary['Sector'].value_counts()
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Stocks per Sector",
            template="plotly_dark",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Data Table")
    st.dataframe(summary, use_container_width=True)

if __name__ == "__main__":
    main()
