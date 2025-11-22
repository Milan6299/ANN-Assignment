clean_lines = []
with open("vehicle2.dat", "r") as f:
    for line in f:
        if line.lower().startswith("@inputs") or line.lower().startswith("@outputs"):
            continue        # skip unsupported lines
        clean_lines.append(line)

with open("vehicle_clean.arff", "w") as f:
    f.writelines(clean_lines)
