from pathlib import Path

data_dir = "./data"

absolute_data_dir = Path(data_dir).resolve()
print("--- absolute data path ---")
print(absolute_data_dir)

folder_list = list(absolute_data_dir.glob("**"))
print("--- Displaying all folders underneath ---")
for folder in folder_list:
    print(str(folder))

png_num = 0
for folder in folder_list:
    imgs_path = Path(folder)
    png_num += len(list(imgs_path.glob("*.png")))
    
print("--- image num ---")
print(png_num)