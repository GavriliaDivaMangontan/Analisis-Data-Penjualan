import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import numpy as np

# Data Produk (20 entri)
data_produk = {
    'Product_ID': list(range(1, 21)),
    'Product_Name': [f'Produk {chr(65+i)}' for i in range(20)],
    'Category': ['Elektronik', 'Fashion', 'Kecantikan', 'Elektronik', 'Fashion'] * 4,
    'Price': [50000, 20000, 15000, 30000, 10000, 45000, 25000, 35000, 28000, 12000, 70000, 22000, 16000, 37000, 11000, 51000, 26000, 33000, 29000, 13000],
    'Stock': [100, 50, 200, 30, 80, 60, 90, 40, 70, 55, 120, 45, 95, 35, 85, 75, 65, 25, 105, 115],
    'Rating': [4.5, 4.0, 3.8, 4.2, 4.1, 4.4, 3.9, 4.3, 4.0, 3.7, 4.6, 4.1, 3.8, 4.2, 4.0, 4.5, 3.9, 4.3, 4.0, 3.6]
}

# Data Penjualan (20 entri)
data_penjualan = {
    'Transaction_ID': list(range(1001, 1021)),
    'Product_ID': list(range(1, 21)),
    'User_ID': [501, 502, 503, 504, 501, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519],
    'Quantity': [2, 1, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 1, 2],
    'Total_Price': [100000, 20000, 45000, 30000, 20000, 45000, 50000, 105000, 28000, 24000, 210000, 22000, 32000, 37000, 33000, 51000, 52000, 99000, 29000, 26000],
    'Date': ['2023-01-15', '2023-02-10', '2023-03-12', '2023-04-05', '2023-05-22', '2023-06-15', '2023-07-10', '2023-08-12', '2023-09-05', '2023-10-22', '2023-11-15', '2023-12-10', '2024-01-12', '2024-02-05', '2024-03-22', '2024-04-15', '2024-05-10', '2024-06-12', '2024-07-05', '2024-08-22']
}

# Data Pengguna (20 entri)
data_pengguna = {
    'User_ID': list(range(501, 521)),
    'User_Name': [f'User {chr(65+i)}' for i in range(20)],
    'Age': [25, 30, 22, 27, 24, 29, 21, 26, 28, 23, 31, 34, 22, 27, 24, 29, 21, 26, 28, 23],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'Location': ['Jakarta', 'Surabaya', 'Bandung', 'Yogyakarta', 'Medan', 'Jakarta', 'Surabaya', 'Bandung', 'Yogyakarta', 'Medan', 'Jakarta', 'Surabaya', 'Bandung', 'Yogyakarta', 'Medan', 'Jakarta', 'Surabaya', 'Bandung', 'Yogyakarta', 'Medan']
}

# Create DataFrames
df_produk = pd.DataFrame(data_produk)
df_penjualan = pd.DataFrame(data_penjualan)
df_pengguna = pd.DataFrame(data_pengguna)

# Analisis Korelasi antara Harga dan Rating Produk
correlation, p_value = stats.pearsonr(df_produk['Price'], df_produk['Rating'])

# Analisis Chi-Square antara Kategori Produk dan Penjualan
# Menghitung frekuensi penjualan per kategori produk
df_penjualan_kategori = pd.merge(df_penjualan, df_produk[['Product_ID', 'Category']], on='Product_ID')
contingency_table = pd.crosstab(df_penjualan_kategori['Category'], df_penjualan_kategori['Quantity'])
chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)

# Analisis Regresi antara Umur Pengguna dan Jumlah Pembelian
user_sales = df_penjualan.groupby('User_ID')['Quantity'].sum().reset_index()
merged_data = pd.merge(user_sales, df_pengguna, on='User_ID')
X = merged_data[['Age']]
y = merged_data['Quantity']
reg = LinearRegression().fit(X, y)

# Analisis Varians (ANOVA) untuk Perbandingan Harga Berdasarkan Kategori
anova_results = stats.f_oneway(df_produk[df_produk['Category'] == 'Elektronik']['Price'],
                               df_produk[df_produk['Category'] == 'Fashion']['Price'],
                               df_produk[df_produk['Category'] == 'Kecantikan']['Price'])

# Visualisasi

# Scatter Plot: Harga vs. Rating Produk (dengan garis regresi)
plt.figure(figsize=(10, 6))
sns.regplot(data=df_produk, x='Price', y='Rating', scatter_kws={'s':100}, line_kws={'color':'red'}, ci=None)
plt.title(f'Scatter Plot: Harga vs. Rating Produk (Koefisien Korelasi: {correlation:.2f}, P-Value: {p_value:.3f})')
plt.xlabel('Harga (IDR)')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

# Bar Chart: Jumlah Produk per Kategori (dengan nilai chi-square)
plt.figure(figsize=(10, 6))
sns.countplot(data=df_produk, x='Category', palette='viridis')
plt.title(f'Bar Chart: Jumlah Produk per Kategori\nChi-Square Statistic: {chi2:.2f}, P-Value: {chi2_p:.3f}')
plt.xlabel('Kategori')
plt.ylabel('Jumlah Produk')
plt.grid(True)
plt.show()

# Scatter Plot: Umur Pengguna vs. Jumlah Pembelian (dengan garis regresi)
plt.figure(figsize=(10, 6))
sns.regplot(data=merged_data, x='Age', y='Quantity', scatter_kws={'s':100}, line_kws={'color':'red'}, ci=None)
plt.title(f'Scatter Plot: Umur Pengguna vs. Jumlah Pembelian (R-squared: {reg.score(X, y):.2f})')
plt.xlabel('Usia')
plt.ylabel('Jumlah Pembelian')
plt.grid(True)
plt.show()

# Box Plot: Harga Berdasarkan Kategori (dengan nilai F-statistic)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_produk, x='Category', y='Price', palette='viridis')
plt.title(f'Box Plot: Harga Berdasarkan Kategori\nF-Statistic: {anova_results.statistic:.2f}, P-Value: {anova_results.pvalue:.3f}')
plt.xlabel('Kategori')
plt.ylabel('Harga (IDR)')
plt.grid(True)
plt.show()
