import os
import glob
import shutil
from utils import Config
from tqdm import tqdm


def main():
    config = Config()
    data_dir = config.options["data_dir"]
    mode = "train"
    # mode = "valid"
    fractured_dir = os.path.join(data_dir, mode, "FRACTURED")
    unfractured_dir = os.path.join(data_dir, mode, "UNFRACTURED")
    os.makedirs(fractured_dir, exist_ok=True)
    os.makedirs(unfractured_dir, exist_ok=True)

    fractured_files = glob.glob(os.path.join(data_dir, "**/*positive/*"), recursive=True)
    unfractured_files = glob.glob(os.path.join(data_dir, "**/*negative/*"), recursive=True)

    def copy_files(src_files, target_dir):
        with tqdm(total=len(src_files)) as pbar:
            print("Copying fractured images...")
            for i, file in enumerate(src_files):
                target_path = os.path.join(target_dir, str(i)+".png")
                if not os.path.exists(target_path):
                    shutil.copy(file, target_path)
                pbar.update(1)

    copy_files(fractured_files, fractured_dir)
    copy_files(unfractured_files, unfractured_dir)


if __name__ == "__main__":
    main()
