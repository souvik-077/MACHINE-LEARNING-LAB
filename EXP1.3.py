import pandas as pd
data = {
    "Student_Name": ["Vicky", "Subham", "Nikhil", "Souvik", "Soumyodip"],
    "Roll_Number": [101, 102, 103, 104, 105],
    "Marks": [79, 84, 67, 91, 77],
    "Attendance": [88, 25, 46, 89, 81]
}

df = pd.DataFrame(data)

df["Grade"] = ""

for i in range(len(df)):
    if df["Marks"][i] >= 90:
        df["Grade"][i] = "A"
    elif df["Marks"][i] >= 80:
        df["Grade"][i] = "B"
    elif df["Marks"][i] >= 70:
        df["Grade"][i] = "C"
    else:
        df["Grade"][i] = "D"

print(df)
