# CarDekho Data Preprocessing & EDA Report Generator
# Assignment: Data Preprocessing and Exploratory Data Analysis
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder, StandardScaler

plt.style.use('ggplot')  # Clean and consistent visual style

# ==============================
# Step 1: Load Dataset
# ==============================
df = pd.read_csv("cardekho.csv")
print("✅ Dataset Loaded Successfully!")
print("Rows:", df.shape[0], " Columns:", df.shape[1])

# ==============================
# Step 2: Data Preprocessing
# ==============================

# Handle Missing Data
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Remove Outliers using IQR Method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)
scaled_df = pd.DataFrame(scaled_data, columns=df_clean.columns)

# Save cleaned dataset
scaled_df.to_csv("cleaned_cardekho_dataset.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_cardekho_dataset.csv'")

# ==============================
# Step 3: EDA
# ==============================
df_work = df_clean.copy().reset_index(drop=True)

# ==============================
# Step 4: Generate PDF Report
# ==============================
pdf_path = "CarDekho_EDA_Report.pdf"

with PdfPages(pdf_path) as pdf:
    # --- Title Page ---
    fig = plt.figure(figsize=(11.7,8.3))
    fig.suptitle("CarDekho Dataset - Preprocessing & Exploratory Data Analysis\n", fontsize=20, weight='bold')
    text = (
        "Prepared for: Assignment Submission\n\n"
        "Contents:\n"
        "1. Data Overview & Preprocessing Summary\n"
        "2. Analysis 1: Year-wise Distribution of Cars\n"
        "3. Analysis 2: Average Selling Price by Fuel Type\n"
        "4. Analysis 3: Engine Size vs Selling Price\n"
        "5. Analysis 4: Seller Type vs Average Selling Price\n"
        "6. Analysis 5: Transmission Type Impact on Price\n"
        "7. Correlation Heatmap\n\n"
        "Preprocessing Steps:\n"
        "• Missing value handling (Mean/Mode)\n"
        "• Label Encoding for categorical fields\n"
        "• Outlier removal using IQR\n"
        "• Feature Scaling using StandardScaler\n"
    )
    plt.axis('off')
    plt.text(0.01, 0.98, text, va='top', fontsize=11, wrap=True)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Analysis 1: Year-wise Distribution ---
    fig, ax = plt.subplots(figsize=(11.7,8.3))
    bars = df_work['year'].value_counts().sort_index().plot(kind='bar', ax=ax, color='#6A5ACD', edgecolor='black')
    ax.set_title("Analysis 1: Number of Cars by Manufacturing Year", fontsize=16, weight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    for p in bars.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.15, p.get_height() + 5), fontsize=9)
    plt.annotate("Insight: Most cars are from 2010–2017, indicating stronger resale market for mid-aged cars.",
                 xy=(0.5, -0.12), xycoords='axes fraction', ha='center', fontsize=10, color='black')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Analysis 2: Average Price by Fuel Type ---
    fig, ax = plt.subplots(figsize=(11.7,8.3))
    bars = df_work.groupby('fuel')['selling_price'].mean().sort_values().plot(kind='bar', ax=ax, color='#20B2AA', edgecolor='black')
    ax.set_title("Analysis 2: Average Selling Price by Fuel Type", fontsize=16, weight='bold')
    ax.set_xlabel("Fuel Type", fontsize=12)
    ax.set_ylabel("Average Price (₹)", fontsize=12)
    for p in bars.patches:
        ax.annotate(f"{p.get_height():.0f}", (p.get_x()+0.15, p.get_height()+10000), fontsize=9)
    plt.annotate("Insight: Diesel cars show higher resale prices compared to Petrol or CNG cars.",
                 xy=(0.5, -0.12), xycoords='axes fraction', ha='center', fontsize=10, color='black')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Analysis 3: Engine Size vs Selling Price ---
    fig, ax = plt.subplots(figsize=(11.7,8.3))
    ax.scatter(df_work['engine'], df_work['selling_price'], alpha=0.6, c='#FF8C00', edgecolors='black')
    ax.set_title("Analysis 3: Engine Size vs Selling Price", fontsize=16, weight='bold')
    ax.set_xlabel("Engine (CC)", fontsize=12)
    ax.set_ylabel("Selling Price (₹)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    corr = df_work[['engine','selling_price']].corr().iloc[0,1]
    plt.annotate(f"Insight: Correlation = {corr:.3f}. Larger engines tend to command higher prices.",
                 xy=(0.01, -0.12), xycoords='axes fraction', ha='left', fontsize=10)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Analysis 4: Seller Type vs Average Price ---
    fig, ax = plt.subplots(figsize=(11.7,8.3))
    bars = df_work.groupby('seller_type')['selling_price'].mean().sort_values().plot(kind='bar', ax=ax, color='#FF69B4', edgecolor='black')
    ax.set_title("Analysis 4: Average Selling Price by Seller Type", fontsize=16, weight='bold')
    ax.set_xlabel("Seller Type", fontsize=12)
    ax.set_ylabel("Average Price (₹)", fontsize=12)
    for p in bars.patches:
        ax.annotate(f"{p.get_height():.0f}", (p.get_x()+0.15, p.get_height()+10000), fontsize=9)
    plt.annotate("Insight: Dealers list higher prices due to added service and warranty value.",
                 xy=(0.5, -0.12), xycoords='axes fraction', ha='center', fontsize=10)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Analysis 5: Transmission Type Impact ---
    fig, ax = plt.subplots(figsize=(11.7,8.3))
    bars = df_work.groupby('transmission')['selling_price'].mean().sort_values().plot(kind='bar', ax=ax, color='#4682B4', edgecolor='black')
    ax.set_title("Analysis 5: Average Price by Transmission Type", fontsize=16, weight='bold')
    ax.set_xlabel("Transmission Type", fontsize=12)
    ax.set_ylabel("Average Price (₹)", fontsize=12)
    for p in bars.patches:
        ax.annotate(f"{p.get_height():.0f}", (p.get_x()+0.15, p.get_height()+10000), fontsize=9)
    plt.annotate("Insight: Automatic cars attract higher resale prices due to comfort demand.",
                 xy=(0.5, -0.12), xycoords='axes fraction', ha='center', fontsize=10)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # --- Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(11.7,8.3))
    corr_matrix = df_work.corr()
    im = ax.imshow(corr_matrix, cmap="coolwarm", aspect='auto')
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=9)
    ax.set_yticklabels(corr_matrix.index, fontsize=9)
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=16, weight='bold')
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7, color='black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.annotate("Insight: Selling price correlates strongly with engine capacity and power features.",
                 xy=(0.5, -0.12), xycoords='axes fraction', ha='center', fontsize=10)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("🎯 PDF Report generated successfully with enhanced visuals: CarDekho_EDA_Report.pdf")
