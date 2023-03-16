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

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to the directory where you want to start searching for .rst files
    src_directory = 'Notion_DB'
    copy_rst_files_to_md(src_directory)

