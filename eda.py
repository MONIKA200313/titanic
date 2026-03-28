# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Load dataset
data = pd.read_csv('C:/Users/dell/OneDrive/Desktop/titanic/dataset.csv')

# -------------------------------
# 1. Basic Data Understanding
# -------------------------------
print("First 5 Rows:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nStatistical Summary:\n", data.describe())

# -------------------------------
# 2. Missing Values
# -------------------------------
print("\nMissing Values:\n", data.isnull().sum())

# Handling missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

print("\nMissing Values After Handling:\n", data.isnull().sum())

# -------------------------------
# 3. Univariate Analysis
# -------------------------------

# Age Distribution
plt.figure()
plt.hist(data['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig('C:/Users/dell/OneDrive/Desktop/titanic/outputs/age_distribution.png')
plt.show()

# Survival Count
plt.figure()
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.savefig('C:/Users/dell/OneDrive/Desktop/titanic/outputs/survival_count.png')
plt.show()

# -------------------------------
# 4. Bivariate Analysis
# -------------------------------

# Survival by Gender
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival by Gender")
plt.savefig('C:/Users/dell/OneDrive/Desktop/titanic/outputs/survival_gender.png')
plt.show()

# Survival by Passenger Class
plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Class")
plt.savefig('C:/Users/dell/OneDrive/Desktop/titanic/outputs/survival_class.png')
plt.show()

# -------------------------------
# 5. Correlation Heatmap
# -------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig('C:/Users/dell/OneDrive/Desktop/titanic/outputs/heatmap.png')
plt.show()

# -------------------------------
# 6. Insights (Printed)
# -------------------------------

print("\nKey Insights:")
print("1. Females have higher survival rate than males.")
print("2. Passengers in 1st class survived more than others.")
print("3. Higher fare passengers had better survival chances.")
print("4. Age has some influence but not very strong.")