"""
Week 1 - Data Acquisition, Cleaning and Preprocessing
Dataset: Titanic passenger data (public, Kaggle / DataSci Dojo mirror)
Run:  python week1_pipeline.py
Outputs: figures in ./figures, cleaned data in titanic_cleaned.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

os.makedirs("figures", exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# ------------------------------------------------------------------
# 1. DATA ACQUISITION
# ------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df_raw = pd.read_csv(URL)
    print("Loaded from URL")
except Exception:
    df_raw = pd.read_csv("titanic.csv")      # fallback: file downloaded manually
    print("Loaded from local titanic.csv")
df = df_raw.copy()

# ------------------------------------------------------------------
# 2. INITIAL EXPLORATION
# ------------------------------------------------------------------
print("\n--- SHAPE ---");   print(df.shape)
print("\n--- HEAD ---");    print(df.head())
print("\n--- INFO ---");    df.info()
print("\n--- DESCRIBE (numeric) ---");  print(df.describe().T)
print("\n--- DESCRIBE (object) ---");   print(df.describe(include="object").T)

# ------------------------------------------------------------------
# 3. MISSING VALUES - DETECTION
# ------------------------------------------------------------------
miss = pd.DataFrame({"missing": df.isnull().sum(),
                     "percent": (df.isnull().mean() * 100).round(2)})
miss = miss[miss.missing > 0].sort_values("missing", ascending=False)
print("\n--- MISSING VALUES ---");  print(miss)

plt.figure(figsize=(7, 4))
sns.barplot(x=miss.index, y=miss.percent, color="steelblue")
plt.ylabel("% missing"); plt.title("Missing values by column")
plt.tight_layout(); plt.savefig("figures/fig1_missing.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 4. DUPLICATES AND ERRONEOUS ENTRIES
# ------------------------------------------------------------------
print("\nFull-row duplicates:", df.duplicated().sum())
print("Duplicate PassengerId:", df["PassengerId"].duplicated().sum())

# whitespace / inconsistent text
for c in ["Name", "Sex", "Ticket", "Cabin", "Embarked"]:
    df[c] = df[c].astype("string").str.strip()
df["Sex"] = df["Sex"].str.lower()
df["Embarked"] = df["Embarked"].str.upper()

print("\nSex values:", df["Sex"].unique().tolist())
print("Embarked values:", df["Embarked"].dropna().unique().tolist())
print("Survived values:", sorted(df["Survived"].unique().tolist()))
print("Pclass values:", sorted(df["Pclass"].unique().tolist()))

# range / logic checks
checks = {
    "Age <= 0 or > 100":  ((df.Age <= 0) | (df.Age > 100)).sum(),
    "Fare < 0":           (df.Fare < 0).sum(),
    "Fare == 0":          (df.Fare == 0).sum(),
    "SibSp < 0":          (df.SibSp < 0).sum(),
    "Parch < 0":          (df.Parch < 0).sum(),
}
print("\n--- VALIDITY CHECKS ---")
for k, v in checks.items():
    print(f"{k:22s}: {v}")

# Fare == 0 is implausible for a paying passenger -> treat as missing
df.loc[df.Fare == 0, "Fare"] = np.nan
print("Fare NaN after flagging zeros:", df.Fare.isnull().sum())

# ------------------------------------------------------------------
# 5. FEATURE EXTRACTION NEEDED FOR IMPUTATION
# ------------------------------------------------------------------
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
title_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
df["Title"] = df["Title"].replace(title_map)
common = ["Mr", "Mrs", "Miss", "Master"]
df["Title"] = df["Title"].where(df["Title"].isin(common), "Rare")
print("\nTitle counts:\n", df["Title"].value_counts())

# ------------------------------------------------------------------
# 6. MISSING VALUE TREATMENT
# ------------------------------------------------------------------
# 6a. Embarked (very few missing) -> mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 6b. Age (substantial) -> median within Title x Pclass groups
age_before = df["Age"].describe()
df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(lambda s: s.fillna(s.median()))
df["Age"] = df["Age"].fillna(df["Age"].median())        # safety net for empty groups
print("\nAge before vs after imputation:")
print(pd.concat([age_before, df["Age"].describe()], axis=1, keys=["before", "after"]))

# 6c. Fare (zeros flagged above) -> median by Pclass
df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))

# 6d. Cabin (most values missing) -> keep information as flag + deck, drop raw column
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("U")
df = df.drop(columns=["Cabin"])

print("\nRemaining missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ------------------------------------------------------------------
# 7. OUTLIER DETECTION AND TREATMENT
# ------------------------------------------------------------------
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
sns.boxplot(y=df["Age"], ax=ax[0], color="lightblue");  ax[0].set_title("Age (before)")
sns.boxplot(y=df["Fare"], ax=ax[1], color="lightblue"); ax[1].set_title("Fare (before)")
plt.tight_layout(); plt.savefig("figures/fig2_boxplots_before.png", dpi=150); plt.close()

for col in ["Age", "Fare"]:
    lo, hi = iqr_bounds(df[col])
    n_iqr = ((df[col] < lo) | (df[col] > hi)).sum()
    z = (df[col] - df[col].mean()) / df[col].std()
    n_z = (z.abs() > 3).sum()
    print(f"{col}: IQR bounds=({lo:.2f}, {hi:.2f}) outliers={n_iqr} | |z|>3 outliers={n_z} "
          f"| skew={df[col].skew():.2f}")

# Age outliers are genuine (elderly passengers) -> keep.
# Fare is heavily right-skewed with genuine first-class fares -> keep, but tame with
# winsorising (cap at 99th percentile) and a log transform.
cap = df["Fare"].quantile(0.99)
df["Fare_capped"] = df["Fare"].clip(upper=cap)
df["Fare_log"] = np.log1p(df["Fare"])
print(f"\nFare 99th percentile cap = {cap:.2f}; skew raw={df.Fare.skew():.2f}, "
      f"capped={df.Fare_capped.skew():.2f}, log={df.Fare_log.skew():.2f}")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
sns.histplot(df["Fare"], kde=True, ax=ax[0]);        ax[0].set_title("Fare (raw)")
sns.histplot(df["Fare_capped"], kde=True, ax=ax[1]); ax[1].set_title("Fare (capped at P99)")
sns.histplot(df["Fare_log"], kde=True, ax=ax[2]);    ax[2].set_title("Fare (log1p)")
plt.tight_layout(); plt.savefig("figures/fig3_fare_treatment.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 8. FEATURE ENGINEERING, ENCODING, SCALING
# ------------------------------------------------------------------
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)

df_model = df.drop(columns=["PassengerId", "Name", "Ticket", "Fare", "Fare_capped"])
df_model = pd.get_dummies(df_model, columns=["Embarked", "Title", "Deck"], drop_first=True, dtype=int)

scale_cols = ["Age", "Fare_log", "FamilySize"]
df_model[scale_cols] = StandardScaler().fit_transform(df_model[scale_cols])

print("\nFinal shape:", df_model.shape)
print(df_model.head())
print("Any NaN left?", df_model.isnull().any().any())
df_model.to_csv("titanic_cleaned.csv", index=False)

plt.figure(figsize=(8, 6))
sns.heatmap(df_model[["Survived", "Pclass", "Sex", "Age", "Fare_log", "FamilySize",
                      "IsAlone", "HasCabin"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix (cleaned data)")
plt.tight_layout(); plt.savefig("figures/fig4_corr.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 9. IMPACT OF PREPROCESSING ON A DOWNSTREAM MODEL
# ------------------------------------------------------------------
# Baseline: raw numeric columns, rows with any missing value dropped
base = df_raw[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].dropna()
base_score = cross_val_score(LogisticRegression(max_iter=1000),
                             base.drop(columns="Survived"), base["Survived"], cv=5).mean()

X, y = df_model.drop(columns="Survived"), df_model["Survived"]
clean_score = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean()
print(f"\nBaseline (drop-NaN rows, {len(base)} samples): CV accuracy = {base_score:.3f}")
print(f"Preprocessed ({len(X)} samples):               CV accuracy = {clean_score:.3f}")
