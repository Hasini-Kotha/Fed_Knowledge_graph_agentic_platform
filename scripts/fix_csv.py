"""Fix quoted CSV — remove surrounding quotes and write as proper CSV."""
with open("data/kaggle_sample.csv", "r") as f:
    lines = [line.strip().strip('"') for line in f if line.strip()]

with open("data/kaggle_sample_clean.csv", "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Fixed {len(lines)} lines. First line: {lines[0]}")
