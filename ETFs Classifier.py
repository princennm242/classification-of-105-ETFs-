import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing all the files

etf = pd.read_csv("Anonymized ETFs.csv")
main_assets = pd.read_csv("Main Asset Classes.csv", skiprows=3)
myst1 = pd.read_csv("Mystery Allocation 1.csv", header=None, names=["Date", "Value"])
myst2 = pd.read_csv("Mystery Allocation 2.csv", header=None, names=["Date", "Value"])

if etf.index.name == "Date":
    etf = etf.reset_index()

etf = etf.rename( columns= {etf.columns[0]: "Date"})
etf["Date"]  = pd.to_datetime(etf["Date"], dayfirst=True, errors="coerce")
for c in etf.columns[1:]:
    etf[c] = pd.to_numeric(etf[c], errors="coerce")
etf = etf.dropna()

#print(etf.columns) #les différents ETFs 
etf = etf.set_index("Date")
etf.index = etf.index.normalize()

daily_returns = etf.pct_change().dropna()
print(daily_returns.head())

# Compute mean and std , both daily and annual

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

main_assets = main_assets.drop([0, 1])   # remove bad rows

# Rename column
main_assets = main_assets.rename(columns={main_assets.columns[0]: "Date"})

# Convert date
main_assets["Date"] = pd.to_datetime(main_assets["Date"], dayfirst=True, errors="coerce")

# Convert numeric values
for c in main_assets.columns[1:]:
    main_assets[c] = pd.to_numeric(main_assets[c], errors="coerce")

# Drop rows with missing values
main_assets = main_assets.dropna()

# Set index
main_assets = main_assets.set_index("Date")
main_assets.index = main_assets.index.normalize()

print("Identicals Index :", etf.index.equals(main_assets.index))
print("Identicals Columns :", etf.columns.equals(main_assets.columns))

print("Missing Values in etf :", etf.isna().sum().sum())
print("Missing Values in main_assets :", main_assets.isna().sum().sum())

main_returns = main_assets.pct_change().dropna()
print(main_returns.head())

# Align the two datasets on the same dates
common_dates = daily_returns.index.intersection(main_returns.index)

etf_ret  = daily_returns.loc[common_dates]
main_ret = main_returns.loc[common_dates]

print(etf_ret.shape, main_ret.shape)

corr_table = pd.DataFrame(
    index=etf_ret.columns,
    columns=main_ret.columns,
    dtype=float
)

for asset in main_ret.columns:
    corr_table[asset] = etf_ret.corrwith(main_ret[asset])

print(corr_table)

best_class = corr_table.idxmax(axis=1)
best_corr  = corr_table.max(axis=1)

etf_class_map = pd.DataFrame({
    "Best_Asset_Class": best_class,
    "Correlation": best_corr
})
print(etf_class_map)

counts = etf_class_map["Best_Asset_Class"].value_counts()
print(counts)

risk_stats = pd.DataFrame({
    "min" : summary["Std Annual"].min(),
    "max" : summary["Std Annual"].max(),
    "mean": summary["Std Annual"].mean(),
    "median": summary["Std Annual"].median(),
    "25%": summary["Std Annual"].quantile(0.25),
    "75%": summary["Std Annual"].quantile(0.75)
}, index=["Annual Volatility"])

print(risk_stats.T)

plt.figure(figsize=(6,4))
plt.boxplot(summary["Std Annual"], vert=True)
plt.title("Boxplot of Annual Volatility (Std Annual)")
plt.ylabel("Annual Volatility")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# If 'Date' is currently the index, reset it to be a column
if myst1.index.name == 'Date':
    myst1 = myst1.reset_index()
# Convert date column to datetime
myst1["Date"] = pd.to_datetime(myst1["Date"], dayfirst=True, errors="coerce")

# Set Date column as index
myst1 = myst1.set_index("Date")

# Convert the value column (which is now the index) to numeric
# Access the single data column by its position [0]
myst1[myst1.columns[0]] = pd.to_numeric(myst1[myst1.columns[0]], errors="coerce")


myst1 = myst1.dropna()

myst1_returns = myst1.pct_change().dropna()

print(myst1_returns.head())

# Calculate daily and annual mean and std for myst1_returns
mean_daily_myst1 = myst1_returns.mean()
std_daily_myst1 = myst1_returns.std()

mean_annual_myst1 = mean_daily_myst1 * 252
std_annual_myst1 = std_daily_myst1 * np.sqrt(252)

# Print the results
print("Mysterious Allocation 1 Performance:")
print(f"Mean Daily Return: {mean_daily_myst1.iloc[0]:.6f}")
print(f"Standard Deviation Daily Return: {std_daily_myst1.iloc[0]:.6f}")
print(f"Mean Annual Return: {mean_annual_myst1.iloc[0]:.6f}")
print(f"Standard Deviation Annual Return: {std_annual_myst1.iloc[0]:.6f}")

# If 'Date' is currently the index, reset it to be a column
if myst2.index.name == 'Date':
    myst2 = myst2.reset_index()

# Now 'Date' should always be a column
# Convert date column to datetime
myst2["Date"] = pd.to_datetime(myst2["Date"], dayfirst=True, errors="coerce")

# Drop rows with missing dates or values
myst2 = myst2.dropna()

# Set Date as index
myst2 = myst2.set_index("Date")

# Convert Value to numeric
# The value column is named 'Value' as per initial load
myst2["Value"] = pd.to_numeric(myst2["Value"], errors="coerce")

# Drop rows with missing values after numeric conversion
myst2 = myst2.dropna()

# Compute returns
myst2_returns = myst2.pct_change().dropna()

# Performance stats
mean_daily_myst2 = myst2_returns["Value"].mean()
std_daily_myst2 = myst2_returns["Value"].std()

mean_annual_myst2 = mean_daily_myst2 * 252
std_annual_myst2 = std_daily_myst2 * np.sqrt(252)

print("Mysterious Allocation 2 Performance:")
print(f"Mean Daily Return: {mean_daily_myst2:.6f}")
print(f"Std Daily Return: {std_daily_myst2:.6f}")
print(f"Mean Annual Return: {mean_annual_myst2:.6f}")
print(f"Std Annual Return: {std_annual_myst2:.6f}")

# Correlation Mystery 1 vs ETFs

# Align the mystery allocation 1 and etf returns on common dates
common_dates_m1_etf = myst1_returns.index.intersection(daily_returns.index)
myst1_returns_aligned = myst1_returns.loc[common_dates_m1_etf]
daily_returns_aligned_m1 = daily_returns.loc[common_dates_m1_etf]

corr_m1 = daily_returns_aligned_m1.corrwith(myst1_returns_aligned["Value"])
top_m1 = corr_m1.sort_values(ascending=False).head(10)

# Correlation Mystery 2 vs ETFs

# Align the mystery allocation 2 and etf returns on common dates
common_dates_m2_etf = myst2_returns.index.intersection(daily_returns.index)
myst2_returns_aligned = myst2_returns.loc[common_dates_m2_etf]
daily_returns_aligned_m2 = daily_returns.loc[common_dates_m2_etf]

corr_m2 = daily_returns_aligned_m2.corrwith(myst2_returns_aligned["Value"])
top_m2 = corr_m2.sort_values(ascending=False).head(10)

asset_lookup = etf_class_map["Best_Asset_Class"].to_dict()

def print_mystery_interpretation(name, top_corr, asset_lookup):
    print(f"\nTop ETFs correlated with {name}:\n")
    for etf, corr in top_corr.items():
        asset_class = asset_lookup.get(etf, "Unknown Asset Class")
        print(f"{etf:<6}   {asset_class:<25}   Correlation: {corr:.4f}")

print_mystery_interpretation("Mystery Allocation 1", top_m1, asset_lookup)

print_mystery_interpretation("Mystery Allocation 2", top_m2, asset_lookup)

