import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import zscore
class LossAversionEngine:
    def __init__(self, ticker: str, start_date: str, end_date: str, lookback_window: int = 20):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback_window
        self.data = None
    def fetch_data(self):
        print(f"[*] Ingesting data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if df.empty:
            raise ValueError(f"No data returned for ticker {self.ticker}")
        
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        self.data = df.copy()
        return self
    def calculate_behavioral_metrics(self, half_life_days: float = 3.0):
        print("[*] Calibrating biological loss-aversion metrics...")
        df = self.data
        
        # Extract flat 1D series data to prevent structure conflicts
        close_series = df['Close'].squeeze()
        volume_series = df['Volume'].squeeze()
        
        df['Returns'] = close_series.pct_change()
        df['Vol_Roll'] = df['Returns'].rolling(window=self.lookback).std()
        df['Vol_ZScore'] = zscore(volume_series.fillna(volume_series.mean()))
        df['Negative_Return_Mask'] = np.where(df['Returns'] < 0, 1, 0)
        
        df['Raw_Panic_Stimulus'] = np.abs(df['Returns']) * df['Vol_ZScore'] * df['Negative_Return_Mask']
        df['Raw_Panic_Stimulus'] = df['Raw_Panic_Stimulus'].fillna(0).clip(lower=0)

        decay_constant = np.log(2) / half_life_days
        amygdala_stress = np.zeros(len(df))
        current_stress = 0.0
         # Pull raw values to ensure loop efficiency
        panic_stim_values = df['Raw_Panic_Stimulus'].values
        for i in range(len(df)):
            current_stress *= np.exp(-decay_constant)
            current_stress += panic_stim_values[i]
            amygdala_stress[i] = current_stress
            
        df['Amygdala_Stress_Score'] = amygdala_stress
        return self
    def generate_signals(self, panic_threshold_z: float = 2.0):
        print("[*] Filtering panic extremes for institutional entry signals...")
        df = self.data
        rolling_mean = df['Amygdala_Stress_Score'].rolling(window=self.lookback * 3, min_periods=1).mean()
        rolling_std = df['Amygdala_Stress_Score'].rolling(window=self.lookback * 3, min_periods=1).std()
        df['Stress_ZScore'] = (df['Amygdala_Stress_Score'] - rolling_mean) / rolling_std.replace(0, np.nan)
        df['Stress_ZScore'] = df['Stress_ZScore'].fillna(0)
        df['Buy_Signal'] = np.where(df['Stress_ZScore'] > panic_threshold_z, 1, 0)
        return df
if __name__ == "__main__":
    # Automatically tracks from Jan 2025 up to today's exact date in October 2026
    today_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    engine = LossAversionEngine(ticker="SPY", start_date="2025-01-01", end_date=today_date)
    
    processed_data = (engine
                      .fetch_data()
                      .calculate_behavioral_metrics(half_life_days=4.0)
                      .generate_signals(panic_threshold_z=2.2))
    
    signals = processed_data[processed_data['Buy_Signal'] == 1]
    
    if len(signals) == 0:
        print(f"\n[-] Analysis complete up to {today_date}. No active panic peaks detected right now. Markets are behaving rationally.")
    else:
        print(f"\n[+] Processing Complete. Found {len(signals)} Modern Capitulation Buy Signals:")
        print(signals[['Close', 'Amygdala_Stress_Score', 'Stress_ZScore']].tail(10))
