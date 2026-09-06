# 🚢 Titanic Survival Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-red)

## 📖 Project Overview

The **Titanic Survival Prediction** project is a Machine Learning project based on the famous Titanic passenger dataset.

The dataset contains information about Titanic passengers, including:

* Passenger class
* Age
* Gender
* Fare
* Number of siblings/spouses
* Number of parents/children
* Embarkation port
* Survival status

The main objective of this project is to analyze passenger information and prepare the data for a **Machine Learning classification model** that predicts whether a passenger survived the Titanic disaster.

The target variable is **`Survived`**:

* `0` → Passenger did not survive
* `1` → Passenger survived

---

# 🎯 Project Objective

The objective of this project is to:

1. Understand the Titanic dataset.
2. Clean the raw data.
3. Handle missing values.
4. Remove unnecessary columns.
5. Perform Exploratory Data Analysis (EDA).
6. Identify important patterns and relationships.
7. Perform feature engineering.
8. Encode categorical variables.
9. Transform and scale numerical features.
10. Prepare the final dataset for Machine Learning.
11. Develop a classification model for survival prediction.

The project is a **Supervised Learning → Binary Classification** problem.

---

# 🧠 Machine Learning Problem

### Problem Type

**Binary Classification**

### Target Variable

```text
Survived
```

### Target Classes

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Did Not Survive |
| `1`   | Survived        |

The model learns from historical passenger information and predicts the survival class of a passenger.

---

# 📂 Dataset

The project uses:

```text
Titanic-Dataset.csv
```

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

df.head()
```

The dataset contains passenger-level information used for analysis and machine learning.

---

# 🛠️ Technologies and Libraries

The following Python libraries are used:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
```

Scikit-learn is used for machine learning preprocessing:

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

### Main Technologies

* 🐍 Python
* 📊 Pandas
* 🔢 NumPy
* 📈 Matplotlib
* 📉 Seaborn
* 🤖 Scikit-learn
* 📓 Jupyter Notebook / VS Code

---

# 📋 Project Workflow

The project follows the following Machine Learning workflow:

```text
Dataset
   ↓
Problem Definition
   ↓
Data Collection
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Feature Encoding
   ↓
Feature Transformation
   ↓
Feature Scaling
   ↓
Final Dataset
   ↓
Machine Learning Model
   ↓
Prediction
   ↓
Evaluation
```

---

# 1️⃣ Problem Definition

The problem is to predict whether a Titanic passenger survived or did not survive based on passenger characteristics.

Example input features include:

* Age
* Sex
* Passenger Class
* Fare
* SibSp
* Parch
* Embarked

The target is:

```text
Survived
```

This makes the problem a **binary classification problem**.

---

# 2️⃣ Data Collection

The Titanic dataset is collected in CSV format and loaded into a Pandas DataFrame.

```python
df = pd.read_csv("Titanic-Dataset.csv")

df
```

Basic dataset inspection is performed using:

```python
df.head()
df.tail()
df.sample()
df.shape
df.columns
```

These commands help understand the structure and size of the dataset.

---

# 3️⃣ Data Understanding

The dataset is examined using:

```python
df.dtypes
df.info()
df.describe()
df.describe(include="all")
```

Additional checks include:

```python
df.notnull()
df.nunique()
df.isnull()
df.duplicated()
df.isnull().sum()
```

Missing-value percentages are also calculated:

```python
missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing_count,
    "Percentage": missing_percentage
})

missing_table = missing_table[
    missing_table["Missing Values"] > 0
]

print(missing_table)
```

Target distribution:

```python
print(df["Survived"].value_counts())

print(
    df["Survived"].value_counts(normalize=True) * 100
)
```

This stage helps identify data types, missing values, duplicate records, unique values, and the target distribution.

---

# 4️⃣ Data Cleaning

Data cleaning is performed before Machine Learning.

The project handles:

* Unnecessary columns
* Duplicate rows
* Missing numerical values
* Missing categorical values

A copy of the dataset is created:

```python
df_clean = df.copy()
```

Unnecessary columns are removed:

```python
columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

df_clean = df_clean.drop(
    columns=columns_to_drop,
    errors="ignore"
)
```

Duplicate rows are removed:

```python
df_clean = df_clean.drop_duplicates()
```

### Numerical Missing Values

Numerical missing values are filled using the median:

```python
numeric_columns = df_clean.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    df_clean[column] = df_clean[column].fillna(
        df_clean[column].median()
    )
```

### Categorical Missing Values

Categorical missing values are filled using the mode:

```python
categorical_columns = df_clean.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df_clean[column] = df_clean[column].fillna(
        df_clean[column].mode()[0]
    )
```

Finally:

```python
print(df_clean.isnull().sum())
```

The goal is to obtain a clean dataset ready for preprocessing.

---

# 5️⃣ Data Preprocessing

The target and features are separated:

```python
target_column = "Survived"

X = df_clean.drop(columns=[target_column])
y = df_clean[target_column]
```

The data is divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Numerical Features

Numerical columns are processed using:

* Median imputation
* StandardScaler

```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Categorical Features

Categorical columns are processed using:

* Most-frequent imputation
* One-hot encoding

```python
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore"
    ))
])
```

### ColumnTransformer

Both transformations are combined:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
```

The preprocessing pipeline is then applied:

```python
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)
```

The project uses an 80/20 train-test split and applies the same learned preprocessing to the test data.

---

# 6️⃣ Exploratory Data Analysis — EDA

Exploratory Data Analysis is performed to understand:

* Survival patterns
* Age distribution
* Passenger class
* Gender
* Fare distribution
* Embarkation ports
* Family relationships
* Feature relationships
* Correlations

---

## 📊 Survival Analysis

### Survival Count

```python
survival = df["Survived"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(survival.index, survival.values)

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()
```

### Survival Percentage

```python
survival = df["Survived"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    survival.values,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%"
)

plt.title("Passenger Survival Percentage")

plt.show()
```

The analysis shows that non-survivors are more numerous than survivors.

---

# 👶 Age Analysis

Age distribution is analyzed using a histogram:

```python
plt.figure(figsize=(7,5))

plt.hist(
    df["Age"].dropna(),
    bins=10
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()
```

The histogram helps understand the distribution of passenger ages.

---

# 🎫 Passenger Class Analysis

```python
pclass = df["Pclass"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(
    pclass.index,
    pclass.values
)

plt.title("Passenger Class Count")
plt.xlabel("Class")
plt.ylabel("Passengers")

plt.show()
```

The analysis shows that **Class 3 contains the largest number of passengers**.

---

# 👨‍👩‍👧 Gender Analysis

```python
gender = df["Sex"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(
    gender.index,
    gender.values
)

plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Passengers")

plt.show()
```

The dataset contains more male passengers than female passengers.

---

# 📦 Box, Violin and Strip Plots

```python
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.boxplot(
    x="Sex",
    y="Age",
    data=df
)
plt.title("Box Plot")

plt.subplot(1,3,2)
sns.violinplot(
    x="Pclass",
    y="Age",
    data=df
)
plt.title("Violin Plot")

plt.subplot(1,3,3)
sns.stripplot(
    x="Pclass",
    y="Age",
    data=df
)
plt.title("Strip Plot")

plt.tight_layout()
plt.show()
```

These plots help understand age distributions, spread, density, and possible outliers.

---

# 💰 Fare Analysis

Fare distribution is analyzed using:

```python
plt.figure(figsize=(8,5))

plt.hist(
    df["Fare"],
    bins=15,
    edgecolor="black"
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.show()
```

The Fare feature is strongly right-skewed, with a small number of passengers paying very high fares.

A Fare box plot is also used:

```python
plt.figure(figsize=(6,5))

plt.boxplot(df["Fare"])

plt.title("Fare Box Plot")
plt.ylabel("Fare")

plt.show()
```

The box plot identifies high-value Fare outliers.

---

# ⚓ Embarked Analysis

```python
embarked = df["Embarked"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(
    embarked.index,
    embarked.values
)

plt.title("Embarked Passengers")
plt.xlabel("Port")
plt.ylabel("Passengers")

plt.show()
```

This visualization compares the number of passengers from different embarkation ports.

---

# 👨‍👩‍👧 Siblings and Spouses Analysis

```python
sib = df["SibSp"].value_counts().sort_index()

plt.figure(figsize=(7,5))

plt.bar(
    sib.index,
    sib.values
)

plt.title("Siblings/Spouses Distribution")
plt.xlabel("SibSp")
plt.ylabel("Passengers")

plt.show()
```

Most passengers had zero siblings or spouses aboard.

---

# 🔗 Multivariate Analysis

The project also analyzes relationships between multiple variables.

### Age by Passenger Class and Survival

```python
plt.figure(figsize=(8,6))

sns.stripplot(
    x="Pclass",
    y="Age",
    hue="Survived",
    data=df,
    jitter=True
)

plt.title(
    "Strip Plot of Age by Passenger Class and Survival"
)

plt.show()
```

### Violin Plot

```python
plt.figure(figsize=(8,6))

sns.violinplot(
    x="Pclass",
    y="Age",
    hue="Survived",
    data=df
)

plt.title(
    "Violin Plot of Age by Passenger Class and Survival"
)

plt.show()
```

These visualizations help identify survival patterns across passenger classes and age groups.

---

# 🔥 Correlation Heatmap

```python
plt.figure(figsize=(10,7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Titanic Dataset Correlation Heatmap"
)

plt.show()
```

The correlation heatmap shows relationships between numerical variables and helps identify positively and negatively related features.

---

# 7️⃣ Feature Engineering

Feature Engineering creates additional useful information from existing features.

The project includes:

### FamilySize

```python
df_fe["FamilySize"] = (
    df_fe["SibSp"] +
    df_fe["Parch"] +
    1
)
```

### IsAlone

```python
df_fe["IsAlone"] = (
    df_fe["FamilySize"] == 1
).astype(int)
```

### FarePerPerson

```python
df_fe["FarePerPerson"] = (
    df_fe["Fare"] /
    df_fe["FamilySize"]
)
```

### Title

```python
df_fe["Title"] = df_fe["Name"].str.extract(
    r",\s*([^.]*)\."
)[0].str.strip()
```

These features can provide additional information about family relationships and passenger characteristics.

---

# 8️⃣ Categorical Encoding

Machine Learning algorithms generally require numerical inputs, so categorical features are converted into numerical representations.

## Label Encoding

For binary categorical data such as `Sex`:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

full_data["Sex"] = le.fit_transform(
    full_data["Sex"]
)
```

---

# One-Hot Encoding

For categorical variables such as `Embarked`:

```python
Embarked_dummies = pd.get_dummies(
    df["Embarked"],
    prefix="Embarked"
)
```

The encoded columns are added to the dataset:

```python
full_data = pd.concat(
    [full_data, Embarked_dummies],
    axis=1
)
```

The project also compares original categorical data with encoded representations.

---

# 9️⃣ Feature Transformation

The Fare feature is positively skewed.

Its statistics can be examined using:

```python
print("Mean:", titanic["Fare"].mean())
print("Median:", titanic["Fare"].median())
print("Skewness:", titanic["Fare"].skew())
```

A log transformation is applied:

```python
titanic_transform["Fare_Log"] = np.log(
    titanic_transform["Fare"] + 1
)
```

This transformation helps reduce the effect of extreme Fare values and makes the distribution more suitable for some Machine Learning algorithms.

---

# 🔟 Feature Scaling

Different scaling techniques are demonstrated.

## Standardization

```python
def standardize(data):
    return (
        data - data.mean()
    ) / data.std()
```

## StandardScaler

```python
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

X_standard = standard_scaler.fit_transform(
    X_train[numeric_features].fillna(
        X_train[numeric_features].median()
    )
)
```

## MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

X_minmax = minmax_scaler.fit_transform(
    X_train[numeric_features].fillna(
        X_train[numeric_features].median()
    )
)
```

MinMaxScaler converts values into a common range, generally from 0 to 1.

## RobustScaler

```python
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()

X_robust = robust_scaler.fit_transform(
    X_train[numeric_features].fillna(
        X_train[numeric_features].median()
    )
)
```

RobustScaler is useful when numerical data contains outliers.

## QuantileTransformer

```python
from sklearn.preprocessing import QuantileTransformer

quantile_scaler = QuantileTransformer(
    output_distribution="normal",
    random_state=42
)

X_quantile = quantile_scaler.fit_transform(
    X_train[numeric_features].fillna(
        X_train[numeric_features].median()
    )
)
```

QuantileTransformer attempts to transform the distribution toward a more normal distribution.

---

# 1️⃣1️⃣ Fare Binning

Fare values are divided into four groups:

```python
fare_bins = [
    0,
    10,
    50,
    100,
    600
]

fare_labels = [
    "Very_Low",
    "Low",
    "Medium",
    "High"
]
```

The bins are created using:

```python
X_train_df["fare_bin"] = pd.cut(
    X_train["fare"],
    bins=fare_bins,
    labels=fare_labels
)
```

Then one-hot encoding is applied:

```python
fare_bin_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)
```

This converts Fare ranges into machine-learning-friendly features.

---

# 1️⃣2️⃣ Final Dataset

The final training and testing shapes are checked:

```python
print(
    "Final Training Shape:",
    X_train_df.shape
)

print(
    "Final Testing Shape:",
    X_test_df.shape
)
```

Missing values are also checked:

```python
print(
    "Final Training Missing Values:",
    X_train_df.isnull().sum().sum()
)

print(
    "Final Testing Missing Values:",
    X_test_df.isnull().sum().sum()
)
```

The target values are checked:

```python
print("y_train:")
print(y_train.head())

print("\ny_test:")
print(y_test.head())
```

The resulting datasets are prepared for Machine Learning modeling.

---

# 📊 EDA Visualizations Included

The project includes several visualizations:

* Survival Count Bar Chart
* Survival Pie Chart
* Age Histogram
* Passenger Class Bar Chart
* Gender Bar Chart
* Box Plot
* Violin Plot
* Strip Plot
* Joint Plot
* Line Chart
* Scatter Plot
* Fare Histogram
* Fare Box Plot
* Embarked Bar Chart
* SibSp Distribution
* Multivariate Strip Plot
* Multivariate Violin Plot
* Point Plot
* KDE Distribution Plot
* Pair Plot
* Survival Pair Plot
* Correlation Heatmap
* Missing Values Heatmap

---

# 🔍 Key EDA Findings

Based on the analysis performed in this project:

* The Titanic dataset contains both numerical and categorical passenger information.
* The target variable is binary: `0` and `1`.
* Non-survivors are more numerous than survivors.
* Class 3 contains the largest number of passengers.
* Male passengers are more numerous than female passengers.
* Most passengers have zero siblings or spouses aboard.
* Fare is strongly right-skewed.
* Fare contains several high-value outliers.
* Passenger age varies across different passenger classes.
* Passenger class, gender, age, and fare provide useful information for survival analysis.

---

# 🧪 Data Preprocessing Techniques

| Technique           | Purpose                             |
| ------------------- | ----------------------------------- |
| Median Imputation   | Handle numerical missing values     |
| Mode Imputation     | Handle categorical missing values   |
| One-Hot Encoding    | Convert categorical variables       |
| Label Encoding      | Encode binary categories            |
| StandardScaler      | Standardize numerical features      |
| MinMaxScaler        | Scale features to 0–1               |
| RobustScaler        | Reduce effect of outliers           |
| QuantileTransformer | Transform feature distribution      |
| Log Transformation  | Reduce skewness                     |
| Fare Binning        | Convert continuous Fare into groups |

---

# 📁 Suggested Project Structure

```text
Titanic/
│
├── Titanic-Dataset.csv
│
├── Titanic.ipynb
│
├── README.md
│
├── requirements.txt
│
└── images/
    ├── survival.png
    ├── age_distribution.png
    ├── fare_distribution.png
    ├── passenger_class.png
    └── correlation_heatmap.png
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Titanic
```

Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

# ▶️ How to Run

### 1. Open the project

Open the project in:

* VS Code
* Jupyter Notebook
* JupyterLab

### 2. Place the dataset

Make sure:

```text
Titanic-Dataset.csv
```

is available in the project directory.

### 3. Run the notebook

Execute the notebook cells from top to bottom.

---

# 🤖 Machine Learning Models

The prepared dataset can be used to train classification algorithms such as:

### Logistic Regression

Used as a simple and interpretable binary classification model.

### K-Nearest Neighbors

Classifies passengers based on similar observations.

### Decision Tree

Creates decision rules based on passenger features.

### Random Forest

Combines multiple decision trees to improve prediction performance.

### Support Vector Machine

Finds a decision boundary between survival classes.

---

# 📈 Model Evaluation

After training the classification models, their performance can be evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC
```

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
```

These metrics can be used to compare different classification models.

---

# 🚀 Future Improvements

The project can be extended by adding:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* SVM
* Cross-validation
* Hyperparameter tuning
* GridSearchCV
* ROC Curve
* Feature importance
* Confusion matrix visualization
* Model comparison
* Final survival prediction system
* Deployment using Flask or Streamlit

---

# 🎓 Learning Outcomes

Through this project, the following Machine Learning concepts are practiced:

* Data Collection
* Data Understanding
* Data Cleaning
* Missing Value Handling
* Exploratory Data Analysis
* Data Visualization
* Feature Engineering
* Feature Selection
* Categorical Encoding
* Feature Transformation
* Feature Scaling
* Train-Test Split
* Preprocessing Pipelines
* ColumnTransformer
* Supervised Learning
* Binary Classification
* Model Evaluation

---

# ⚠️ Important Notes

The project contains multiple demonstrations of preprocessing, encoding, transformation, and scaling techniques for learning purposes.

For a final production model, the preprocessing pipeline should be kept consistent from training through prediction, and transformations should be fitted only on training data to avoid data leakage. The project demonstrates this approach by fitting the preprocessor on training data and transforming the test data using the same fitted preprocessor.

---

# 👨‍💻 Author

**ZARFSHAN ATTIQ KHAN**

**zarfshankahn478@gmail.com**

**Titanic Machine Learning Project**

Developed as a Machine Learning / Data Analysis project using Python and Scikit-learn.

---

# ⭐ Conclusion

The Titanic Survival Prediction project demonstrates a complete Machine Learning data preparation workflow.

Starting from the raw Titanic dataset, the project performs **data understanding, cleaning, missing-value handling, EDA, feature engineering, categorical encoding, feature transformation, and feature scaling**.

The final processed dataset is ready to be used for training and evaluating **binary classification algorithms** to predict passenger survival.

---

## 📌 Project Status

**Status:** 🟢 Data Analysis & Preprocessing Completed

**Next Step:** 🤖 Train and evaluate Machine Learning classification and Regression models.
