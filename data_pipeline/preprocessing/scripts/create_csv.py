import csv

base_path = "TrungDuc/"

file_name = input("Name: ") + ".csv"
full_path = base_path + file_name

data = [
    ["id", "SBP", "DBP", "fs", "start"]
]

print("Enter data for each column:")
id = input("id: ") or "N/A"
sbp = input("SBP: ") or "N/A"
dbp = input("DBP: ") or "N/A"
fs = input("fs: ") or "N/A"
start = input("start: ") or "N/A"

data.append([id, sbp, dbp, fs, start])

with open(full_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f" CSV '{full_path}' created.")
