import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# importing all the files

etf = pd.read_csv(r"C:\Users\princ\Desktop\Python Project\Anonymized ETFs.csv")
main_assets = pd.read_csv(r"C:\Users\princ\Desktop\Python Project\Main Asset Classes.csv")
myst1 = pd.read_csv(r"C:\Users\princ\Desktop\Python Project\Mystery Allocation 1.csv")
myst2 = pd.read_csv(r"C:\Users\princ\Desktop\Python Project\Mystery Allocation 2.csv")

if etf.index.name == "Date":
    etf = etf.reset_index()

etf = etf.rename( columns= {etf.columns[0]: "Date"})
etf["Date"]  = pd.to_datetime(etf["Date"], dayfirst=True, errors="coerce")
for c in etf.columns[1:]:
    etf[c] = pd.to_numeric(etf[c], errors="coerce")
etf = etf.dropna()

etf = etf.set_index("Date")
etf.index = etf.index.normalize()

# Compute mean and std , both daily and annual

daily_returns = etf.pct_change().dropna()
print("Daily Returns Head:\n")
print(daily_returns.head())

mean_daily = daily_returns.mean()
std_daily = daily_returns.std()

mean_annual = mean_daily * 252
std_annual = std_daily * np.sqrt(252)

summary = pd.DataFrame(
    {"Mean Daily": mean_daily,
     "Std Daily": std_daily,
     "Mean Annual": mean_annual,
     "Std Annual": std_annual
})
# sort values of the dataframe using .sort_values(by= "what you want to sort for", ascending= False)
# The ascending is either True if you want from small to big number, False from big to small
classification_daily_performance = summary.sort_values(by="Mean Daily", ascending=False)
classification_annual_performance = summary.sort_values(by="Mean Annual", ascending=False)
classification_daily_risk = summary.sort_values(by="Std Daily", ascending=False)
classification_annual_risk = summary.sort_values(by="Std Annual", ascending=False)

#relationship between etfs
corr = daily_returns.corr()

# remove self-correlation (ETF with itself)
corr_pairs = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack()

# Top 10 most correlated ETF pairs
top_corr = corr_pairs.sort_values(ascending=False).head(10)
print("\nMost correlated pairs:\n")
print(top_corr.to_string())

# 10 least correlated ETF pairs
low_corr = corr_pairs.sort_values(ascending=True).head(10)
print("\nLeast correlated pairs:\n")
print(low_corr.to_string())

# Choose 4–5 ETFs with highest volatility as anchors (often equities/commodities)
anchors = summary.sort_values("Std Daily", ascending=False).head(5).index

group_map = {}
for etf_name in daily_returns.columns:
    best_anchor = corr.loc[etf_name, anchors].idxmax()
    group_map[etf_name] = best_anchor

relationship_groups = pd.Series(group_map, name="Most_Correlated_Anchor")
print("\nRelationship-based groups:\n")
print(relationship_groups.value_counts())

print("\n We can see how some ETF pairs have a correlation coefficient of 0.9999, this means that they basically track and represent the same asset class. In addiction there are some ETF pairs that have a negative correlation and this means that they represent different asset classes and that they move in opposite directions, providing a good and natural hedging solution in the portfolio. This can also reduce portfolio risk.\n")
