import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Gender": [
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male"
    ],
    "StudyHours": [
        5, 4, 7, 3, 6, 5,
        8, 2, 4, 6, 7, 3
    ],
    "Attendance(%)": [
        90, 85, 92, 78, 88, 80,
        95, 75, 86, 89, 91, 82
    ],
    "Marks": [
        82, 75, np.nan, 68, 79, 70,
        88, 60, 73, np.nan, 85, 65
    ]
}

df = pd.DataFrame(data)

df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

df.to_csv("student_performance.csv", index=False)

df = pd.read_csv("student_performance.csv")

print("Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df_clean = df.drop_duplicates()

marks_median = df_clean["Marks"].median()

df_clean["Marks"] = df_clean["Marks"].fillna(marks_median)

print("\nAfter cleaning:", df_clean.shape)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

plt.figure(figsize=(6, 4))
plt.hist(df_clean["Marks"], bins=6, edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Marks Distribution")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.boxplot(df_clean["Marks"])
plt.ylabel("Marks")
plt.title("Marks Boxplot")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(df_clean["StudyHours"], df_clean["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.show()

numeric_columns = [
    "StudyHours",
    "Attendance(%)",
    "Marks"
]

corr_matrix = df_clean[numeric_columns].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

gender_summary = df_clean.groupby("Gender")[
    ["Marks", "Attendance(%)"]
].mean()

print("\nAverage Marks and Attendance by Gender:")
print(gender_summary)

print(
    "\nSaved marks_histogram.png, "
    "marks_boxplot.png, studyhours_vs_marks.png"
)
