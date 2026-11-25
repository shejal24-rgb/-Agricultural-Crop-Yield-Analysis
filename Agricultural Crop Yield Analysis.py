import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("crop_yield.csv")

df = df.drop_duplicates()
df = df.dropna()

df["Yield_Per_Hectare"] = df["Production_Tonnes"] / df["Area_Hectares"]

print("Dataset Overview")
print(df.head())

print("\nCrop Production Summary")
print(df.groupby("Crop")["Production_Tonnes"].sum().sort_values(ascending=False))

print("\nAverage Yield by Crop")
print(df.groupby("Crop")["Yield_Per_Hectare"].mean().sort_values(ascending=False))

print("\nProduction by State")
print(df.groupby("State")["Production_Tonnes"].sum().sort_values(ascending=False))

print("\nSeason-wise Production")
print(df.groupby("Season")["Production_Tonnes"].sum())

print("\nSoil-wise Average Yield")
print(df.groupby("Soil_Type")["Yield_Per_Hectare"].mean())

print("\nCorrelation Analysis")
print(df[["Rainfall_mm", "Temperature_C", "Fertilizer_kg", "Area_Hectares", "Production_Tonnes", "Yield_Per_Hectare"]].corr())

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Crop", y="Production_Tonnes", estimator="sum")
plt.title("Total Production by Crop")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="State", y="Production_Tonnes", estimator="sum")
plt.title("Agricultural Production by State")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Season", y="Yield_Per_Hectare")
plt.title("Crop Yield Distribution by Season")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Rainfall_mm", y="Yield_Per_Hectare", hue="Crop")
plt.title("Rainfall vs Crop Yield")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Fertilizer_kg", y="Production_Tonnes", hue="Season")
plt.title("Fertilizer Usage vs Production")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.heatmap(
    df[["Rainfall_mm", "Temperature_C", "Fertilizer_kg",
        "Area_Hectares", "Production_Tonnes", "Yield_Per_Hectare"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Agricultural Data Correlation")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="Soil_Type", y="Yield_Per_Hectare")
plt.title("Average Crop Yield by Soil Type")
plt.tight_layout()
plt.show()

print("\nKey Insights")
print("Highest production crop:", df.groupby("Crop")["Production_Tonnes"].sum().idxmax())
print("Highest production state:", df.groupby("State")["Production_Tonnes"].sum().idxmax())
print("Highest average yield crop:", df.groupby("Crop")["Yield_Per_Hectare"].mean().idxmax())
print("Average crop yield:", round(np.mean(df["Yield_Per_Hectare"]), 2))