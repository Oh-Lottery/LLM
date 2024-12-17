import pandas as pd
import numpy as np

# Read CSV data
df = pd.read_csv('oh-lottery.csv')

# Count main numbers frequency
main_numbers = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Number 6']].values.ravel()
bonus_numbers = df['Bonus Number'].values

# Combine all numbers
all_numbers = np.concatenate([main_numbers, bonus_numbers])

# Count frequency of each number
number_counts = pd.Series(all_numbers).value_counts()

# Get frequencies of top 6 numbers
top_6_freq = number_counts.head(6).values
# Include all numbers that match the 6th highest frequency
all_top_numbers = number_counts[number_counts >= top_6_freq[-1]]

print("\nMost frequent numbers:")
for num, count in all_top_numbers.items():
    print(f"Number {num}: appeared {count} times")