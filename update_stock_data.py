import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DATA_FOLDER = r"D:\Datasets\Stock trend prediction datasets(N50)"

# Symbol Mapping (Stock Name -> Yahoo Ticker)
# Note: Most NSE stocks need '.NS' suffix
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

def update_stock_data():
    print("Starting Stock Data Update...")
    updated_count = 0
    
    # Iterate over all files in the directory
    for filename in os.listdir(DATA_FOLDER):
        if filename.startswith("N50 Stock data  - ") and filename.endswith(".csv"):
            stock_name = filename.replace("N50 Stock data  - ", "").replace(".csv", "")
            filepath = os.path.join(DATA_FOLDER, filename)
            
            ticker_symbol = SYMBOL_MAP.get(stock_name)
            if not ticker_symbol:
                print(f"Skipping {stock_name}: No ticker symbol found.")
                continue
                
            try:
                # Load existing data
                df = pd.read_csv(filepath)
                
                # Parse Dates (Handle custom format DD/MM/YYYY HH:MM:SS)
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                
                # Get last date
                if not df['Date'].empty:
                    last_date = df['Date'].max()
                else:
                    # Fallback if empty (unlikely)
                    last_date = datetime.now() - timedelta(days=365)
                
                # Check if update is needed (if last date is before today)
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                if last_date >= today:
                    # Already up to date
                    continue
                
                print(f"Updating {stock_name} (Last: {last_date.date()})...")
                
                # Fetch new data from yfinance
                # Start from next day
                start_date = last_date + timedelta(days=1)
                ticker = yf.Ticker(ticker_symbol)
                new_data = ticker.history(start=start_date, interval="1d")
                
                if new_data.empty:
                    print(f"   No new data found for {stock_name}.")
                    continue
                
                # Format new data to match CSV structure
                # CSV Cols: Date, Open, High, Low, Close, Volume
                new_rows = []
                for idx, row in new_data.iterrows():
                    # Format date as DD/MM/YYYY 15:30:00 (Market Close)
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
                    # Append to original CSV (read as text to avoid date parsing issues on save)
                    # Actually, better to append to dataframe and save with specific format
                    
                    # Re-read original as string to preserve exact format if needed, 
                    # OR just save the combined DF with specific date format.
                    # Let's use the DataFrame approach but ensure date format is correct.
                    
                    # Convert original 'Date' back to string format for consistency
                    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y %H:%M:%S')
                    
                    # Concatenate
                    final_df = pd.concat([df, new_df], ignore_index=True)
                    
                    # Save
                    final_df.to_csv(filepath, index=False)
                    print(f"Added {len(new_rows)} new records to {stock_name}.")
                    updated_count += 1
                    
            except Exception as e:
                print(f"Error updating {stock_name}: {e}")

    print(f"\nUpdate Complete! Updated {updated_count} files.")

if __name__ == "__main__":
    update_stock_data()
