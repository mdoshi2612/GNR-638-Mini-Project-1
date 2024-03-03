import argparse
import os
import shutil

def process_file(file_path, root_path, split_path, output_folder):

    with open(split_path, 'r') as file:
        split = [int(line.strip().split()[1]) for line in file]

    image_number = 0
    with open(file_path, 'r') as file:
        for _, line_content in enumerate(file, 1):
            # Process each line as needed
            class_name = line_content.split(' ')[-1].strip()
            class_folder = os.path.join(root_path, class_name)
            train_folder = os.path.join(output_folder, 'train', class_name)
            test_folder = os.path.join(output_folder, 'test', class_name)
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)

            for filename in os.listdir(class_folder):
                if(split[image_number] == 1):
                    shutil.copy(os.path.join(class_folder, filename), train_folder)
                    print("Saved to train_folder")
                else:
                    shutil.copy(os.path.join(class_folder, filename), test_folder)
                    print("Saved to test_folder")
                image_number += 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read a text file line by line.")
    parser.add_argument("--file_path", help="Path to the input text file containing all classes")
    parser.add_argument("--root_path", help="Root path to the dataset")
    parser.add_argument("--split_path", help="Path to the train test split file")
    parser.add_argument("--output_path", help="Output folder path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    process_file(args.file_path, args.root_path, args.split_path, args.output_path)
