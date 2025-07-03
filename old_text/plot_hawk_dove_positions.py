import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
data_path = 'fed_speech_sentiment_analysis.csv'
output_dir = 'plots'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
# Ensure 'date' column is parsed as datetime
df = pd.read_csv(data_path, parse_dates=['date'])

# Verify necessary columns exist
required_cols = {'speaker', 'date', 'hd_mean_score'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Group by speaker and plot
speakers = df['speaker'].dropna().unique()
for sp in speakers:
    sub = df[df['speaker'] == sp].sort_values('date')
    if sub.empty:
        continue

    plt.figure(figsize=(10, 4))
    plt.plot(sub['date'], sub['hd_mean_score'], marker='o', linestyle='-')
    plt.axhline(0, linestyle='--', linewidth=0.8)
    plt.title(f'Hawk–Dove Score Over Time: {sp}')
    plt.xlabel('Date')
    plt.ylabel('Mean Hawk–Dove Score')
    plt.tight_layout()

    # Save each speaker's chart
    filename = f"{sp.replace(' ', '_').replace('/', '_')}_hawk_dove.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"Charts generated for {len(speakers)} speakers in '{output_dir}/'.")
