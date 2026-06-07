# 🛍️ Black Friday Purchase Prediction

Exploratory data analysis and regression modeling to predict individual purchase amounts during Black Friday sales events.

---

## 📂 Dataset

**Source:** [Black Friday Dataset – Kaggle](https://www.kaggle.com/datasets/mehdidag/black-friday)

| Feature | Description |
|---|---|
| `User_ID` | Unique customer identifier |
| `Product_ID` | Unique product identifier |
| `Gender` | Customer gender (M/F) |
| `Age` | Age group (e.g., 0-17, 18-25, ..., 55+) |
| `Occupation` | Occupation code (0–20) |
| `City_Category` | City tier (A, B, C) |
| `Stay_In_Current_City_Years` | Years residing in current city |
| `Marital_Status` | 0 = Single, 1 = Married |
| `Product_Category_1/2/3` | Product category codes |
| `Purchase` | **Target variable** — purchase amount in USD |

- **550,068 rows × 12 columns**
- Missing data: `Product_Category_2` (31.6%) and `Product_Category_3` (69.7%) → dropped during preprocessing

---

## 📁 Project Structure

```
black-friday/
├── data/
│   └── BlackFriday.csv
├── notebook/
│   └── black_friday_analysis.ipynb
└── README.md
```

---

## 📊 Analysis Pipeline

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics for `Purchase` (mean: ~$9,264, std: ~$5,023, range: $12–$23,961)
- Distribution analysis: right-skewed with positive asymmetry; CV ≈ 0.54 (moderate dispersion)
- Univariate analysis: histograms and boxplots for all quantitative and qualitative variables
- Bivariate analysis: scatter plots and Pearson correlation between `Purchase` and all features

### 2. Feature Engineering
- One-hot encoding for categorical variables: `Age`, `Gender`, `City_Category`, `Stay_In_Current_City_Years`
- Created composite key `userxproduct` (`User_ID` + `Product_ID`) — later dropped as non-predictive
- Final feature set: **20 predictors**

### 3. Correlation Findings
All numeric features showed near-zero correlation with `Purchase`:

| Variable | Correlation with Purchase |
|---|---|
| `Occupation` | +0.021 |
| `User_ID` | +0.005 |
| `Marital_Status` | −0.000 |
| `Product_Category_3` | −0.022 |
| `Product_Category_2` | −0.210 |
| `Product_Category_1` | −0.344 |

No variable exceeded the |0.5| threshold. This is a structurally weak regression problem.

### 4. Outlier Detection
- Standardized `Purchase` using `StandardScaler`
- Custom outlier function (threshold: ±6 standard deviations)
- Result: **0 outliers detected**

### 5. Train/Test Split
- 80/20 split (stratified by random seed)
- Training set: 440,054 rows | Test set: 110,014 rows
- Mean `Purchase`: full = 9,264 / train = 9,271 / test = 9,236 ✅ well balanced

---

## 🤖 Models & Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3,598.7 | 4,696.6 | 0.126 |
| Decision Tree Regressor | 2,173.9 | 2,977.1 | 0.649 |
| Random Forest Regressor | 2,163.7 | 2,947.8 | 0.656 |
| XGBoost Regressor | 2,182.7 | 2,934.4 | 0.659 |

**Best model:** XGBoost (lowest RMSE, highest R²)  
Linear Regression performed poorly (R² = 0.13), consistent with the weak linear correlations observed during EDA.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `StandardScaler`, `train_test_split`)
- XGBoost

---

## ▶️ How to Run

```bash
git clone https://github.com/sesquivelc/black-friday.git
cd black-friday
pip install -r requirements.txt
jupyter notebook notebook/black_friday_analysis.ipynb
```

---

## 💡 Key Takeaways

- Purchase amount is **not strongly linearly correlated** with any available feature — the problem is inherently noisy.
- Tree-based models (Random Forest, XGBoost) captured non-linear interactions significantly better than linear regression (R² ~0.66 vs ~0.13).
- `Product_Category_1` had the strongest (negative) correlation with `Purchase`, suggesting lower-numbered categories tend toward higher purchase amounts.
- An R² of ~0.66 leaves room for improvement — candidate approaches include user-level aggregation, interaction features, or neural collaborative filtering.
