from datasets import load_dataset
import os

os.makedirs("Data_Training/sources/languages", exist_ok=True)
os.makedirs("Data_Training/sources/programming", exist_ok=True)

def save_texts(dataset_name, out_file, n=2000, data_dir=None, text_col="content"):
    print(f"Downloading {dataset_name} {data_dir or ''} ...")
    kwargs = {"split": "train", "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    # لبعض الداتا ست اسم العمود text مو content
    if dataset_name == "HuggingFaceFW/fineweb-edu":
        text_col = "text"
        kwargs = {"split": "train", "streaming": True}

    ds = load_dataset(dataset_name, **kwargs)
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for row in ds:
            t = row.get(text_col) or row.get("text") or row.get("content") or ""
            if not t:
                continue
            for line in str(t).split("\n"):
                line = line.strip()
                if 40 < len(line) < 600:
                    f.write(line + "\n")
                    count += 1
                    if count >= n:
                        print(f"saved {count} -> {out_file}")
                        return
    print(f"saved {count} -> {out_file}")

# 1- انقلش + عربي
save_texts("HuggingFaceFW/fineweb-edu", "Data_Training/sources/languages/en_ar.txt", 2000)

# 2- كود C++ / C / Assembly - الحين مفتوح لك
save_texts("bigcode/starcoderdata", "Data_Training/sources/programming/code_cpp_assembly.txt", 2000, data_dir="python")

print("DONE")