import pandas as pd

data = {
    "Student_Name": ["Vicky", "Subham", "Nikhil", "Souvik", "Soumyodip"],
    "Roll_Number": [101, 102, 103, 104, 105],
    "Marks": [79, 84, 67, 91, 77],
    "Attendance": [88, 25, 46, 89, 81]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

print("\nStudents scoring above 80:")
print(df[df["Marks"] > 80])