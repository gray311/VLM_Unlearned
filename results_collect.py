import os
import shutil

def find_eval_log_directories(root_dir):
    eval_log_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'eval_results' in dirnames:
            eval_log_dirs.append(os.path.join(dirpath, 'eval_results'))
    return eval_log_dirs

def copy_eval_log_contents(eval_log_dirs, destination):
    for eval_log_dir in eval_log_dirs:
        tmp = eval_log_dir.split("/")[-2].strip(" ")
        dist = os.path.join(destination, tmp)
        os.makedirs(dist, exist_ok=True)
        for item in os.listdir(eval_log_dir):
            s = os.path.join(eval_log_dir, item)
            d = os.path.join(dist, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

def main():
    root_dir = './models'  # The root directory to search
    destination = './results'  # The destination directory

    print("Searching for eval_log directories...")
    eval_log_dirs = find_eval_log_directories(root_dir)
    print(eval_log_dirs)
    if not eval_log_dirs:
        print("No eval_log directories found.")
        return

    # print(f"Found {len(eval_log_dirs)} eval_log directories. Copying contents...")
    copy_eval_log_contents(eval_log_dirs, destination)
    # print(f"All eval_log contents have been successfully copied to {destination}")

if __name__ == "__main__":
    main()
