from pathlib import Path

data_dir = "./data"
data_dir_path = Path(data_dir).resolve()
print("--- Displaying the absolute path ---")
print(data_dir_path)

# dataディレクトリ以下のすべてのファイルを取得
file_list = list(data_dir_path.glob("*"))
print("--- Displaying all files underneath ---")
for file in file_list:
    print(str(file))

base_dir = Path("/base_dir")
subdirectory_name = "data_dir"

new_path = base_dir / subdirectory_name

print("--- joint paths ---")
print(str(new_path))
for file in file_list:
    if file.exists():
        print(f"The file '{file}' exists.")
    else:
        print(f"The file '{file}' does not exist.")

# 作成したいディレクトリのパス
dir_path = Path("./target_dir")

# ディレクトリを作成
dir_path.mkdir(parents=True, exist_ok=True)
