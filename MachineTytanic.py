import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox, scrolledtext


def analyze_data():
    url = "https://github.com/jbryer/CompStats/raw/master/Data/titanic3.csv"
    titanic_df = pd.read_csv(url)

    results_text.insert(tk.END, titanic_df.info())
    results_text.insert(tk.END, "\n\n")
    results_text.insert(tk.END, titanic_df.describe())
    results_text.insert(tk.END, "\n\n")
    results_text.insert(tk.END, titanic_df.isnull().sum())


def feature_engineering(titanic_df):
    titanic_df['title'] = titanic_df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    age_fill_values = {
        'Miss': titanic_df[(titanic_df['title'] == 'Miss') & (titanic_df['parch'] == 0)]['age'].mean(),
        'Master': titanic_df[titanic_df['title'] == 'Master']['age'].mean(),
        'Mr': titanic_df[titanic_df['title'] == 'Mr']['age'].mean(),
        'Mrs': titanic_df[titanic_df['title'] == 'Mrs']['age'].mean(),
        'Dr': titanic_df[titanic_df['title'] == 'Dr']['age'].mean()
    }

    if 'Ms' in titanic_df['title'].unique():
        age_fill_values['Ms'] = titanic_df[titanic_df['title'] == 'Ms']['age'].mean()

    titanic_df['age'] = titanic_df.apply(
        lambda row: age_fill_values[row['title']] if pd.isnull(row['age']) else row['age'], axis=1)
    titanic_df['family_size'] = titanic_df['parch'] + titanic_df['sibsp'] + 1
    titanic_df['age_range'] = pd.cut(titanic_df['age'], bins=[0, 6, 12, 18, titanic_df['age'].max()],
                                     labels=['baby', 'child', 'teenager', 'adult'])
    titanic_df['mpc'] = titanic_df['age'] * titanic_df['pclass']

    return titanic_df


def metadata_editing(titanic_df):
    titanic_df['survived'] = titanic_df['survived'].astype('category')
    titanic_df['pclass'] = titanic_df['pclass'].astype('category')
    titanic_df['sex'] = titanic_df['sex'].astype('category')
    titanic_df['embarked'] = titanic_df['embarked'].astype('category')
    titanic_df['fare'] = titanic_df['fare'].astype(float)

    embarked_mode = titanic_df['embarked'].mode()[0]
    titanic_df['embarked'] = titanic_df['embarked'].fillna(embarked_mode)

    return titanic_df


def trim_outliers(titanic_df):
    age_mean = titanic_df['age'].mean()
    titanic_df.loc[titanic_df['age'] > 67, 'age'] = age_mean
    z_scores = stats.zscore(titanic_df['fare'])
    threshold = 3
    titanic_df.loc[abs(z_scores) > threshold, 'fare'] = np.nan
    fare_mean = titanic_df['fare'].mean()
    titanic_df['fare'] = titanic_df['fare'].fillna(fare_mean)

    return titanic_df


def normalize_data(titanic_df):
    numerical_cols = ['age', 'fare']
    scaler = MinMaxScaler()
    titanic_df[numerical_cols] = scaler.fit_transform(titanic_df[numerical_cols])

    return titanic_df


def perform_classification():
    url = "https://github.com/jbryer/CompStats/raw/master/Data/titanic3.csv"
    titanic_df = pd.read_csv(url)

    titanic_df = feature_engineering(titanic_df)
    titanic_df = metadata_editing(titanic_df)
    titanic_df = trim_outliers(titanic_df)
    titanic_df = normalize_data(titanic_df)

    features = ["sex", "age", "pclass", "fare"]
    target = "survived"
    X = titanic_df[features]
    y = titanic_df[target]

    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results_text.insert(tk.END, f"Accuracy: {accuracy}\n")
    results_text.insert(tk.END, "Classification Report:\n")
    results_text.insert(tk.END, classification_report(y_test, y_pred))
    messagebox.showinfo("Classification Completed",
                        f"Accuracy: {accuracy}\nCheck the results text area for the full report.")


root = tk.Tk()
root.title("Titanic Data Analysis and Classification")

frame = tk.Frame(root)
frame.pack(pady=10)

analyze_button = tk.Button(frame, text="Analyze Data", command=analyze_data)
analyze_button.grid(row=0, column=0, padx=10, pady=10)

classify_button = tk.Button(frame, text="Perform Classification", command=perform_classification)
classify_button.grid(row=0, column=1, padx=10, pady=10)

results_text = scrolledtext.ScrolledText(root, width=100, height=20)
results_text.pack(padx=10, pady=10)

root.mainloop()
