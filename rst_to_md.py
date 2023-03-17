import os
import shutil

def copy_rst_files_to_md(src_directory):
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            if file.endswith('.rst'):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(root, file[:-4] + '.md')
                shutil.copy(src_file, dest_file)
                print(f'Copied: {src_file} to {dest_file}')
                os.remove(src_file)
                print(f'Deleted: {src_file}')

def get_total_md_size(src_directory):
    total_size = 0
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    return total_size / (1024)  # Convert bytes to kilobytes

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to the directory where you want to start searching for .rst files
    src_directory = 'Repo_DB'
    copy_rst_files_to_md(src_directory)
    total_md_size = get_total_md_size(src_directory)
    print(f'Total size of .md files in {src_directory}: {total_md_size:.2f} KB')
