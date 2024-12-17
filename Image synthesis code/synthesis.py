import cv2
import os
import json
import random
import numpy as np
from collections import defaultdict

from pipeline import main, autogen_framework, encode_image

image_size = (224, 224, 3)

dataset_folder = "ImageNet_subset"

patch_size = 112

api_key = os.environ.get("OPENAI_API_KEY")

seed = 42
np.random.seed(seed)

def read_files_from_folders(folder_path, output_file="combination.txt", num_files=2):
    if os.path.exists(output_file):
        pairs = []
        with open(output_file, "r") as f:
            for line in f:
                folder1, file1, folder2, file2 = line.strip().split(" ")
                pairs.append((folder1, file1, folder2, file2))
        return pairs

    subfolder_to_files = {}
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            if len(files) >= num_files:
                subfolder_to_files[subfolder] = random.sample(files, num_files)
    
    all_combinations = []
    subfolders = list(subfolder_to_files.keys())
    for i in range(0, len(subfolders), 2):
        j = i + 1
        folder1, folder2 = subfolders[i], subfolders[j]
        file1, file2 = subfolder_to_files[folder1], subfolder_to_files[folder2]
        all_combinations.extend([(f"{folder1}", f"{file1[0]}", f"{folder2}", f"{file2[0]}"),
                                    (f"{folder1}",f"{file1[1]}", f"{folder2}",f"{file2[1]}")])
    
    with open(output_file, "w") as f:
        for pair in all_combinations:
            f.write(f"{pair[0]} {pair[1]} {pair[2]} {pair[3]}\n")
    return all_combinations


def processing(image_size, dataset_folder):
    train_folder = os.path.join(dataset_folder, "train")
    label_file = os.path.join(dataset_folder, "label.txt")
    labels = []
    label_to_index = {}
    filename2label = []
    filename2onehot = {}
    with open(label_file, "r") as f:
        for line in f:
            filename, label = line.strip().split(" ", 1)
            if filename in os.listdir(train_folder):
                labels.append((os.path.join(train_folder, filename), label))
                if label not in label_to_index:
                    label_to_index[label] = len(label_to_index)
                    filename2label.append((filename, label))
    num_classes = len(label_to_index)
    label_to_onehot = {label: np.eye(num_classes)[index].tolist() for label, index in label_to_index.items()}
    filename2onehot = {filename: label_to_onehot[label] for filename, label in filename2label}

    if not os.path.exists("label2onehot.json"):
        with open("label2onehot.json", "w") as f:
            json.dump(label_to_onehot, f)
    else:
        label_to_onehot = json.load(open("label2onehot.json"))

    if not os.path.exists("filename2onehot.json"):
        with open("filename2onehot.json", "w") as f:
            json.dump(filename2onehot, f)

    subfolders = [os.path.join(train_folder, d) for d in os.listdir(train_folder)]

    folder_to_files = defaultdict(list)
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            folder_to_files[subfolder].append(file)

    folder_to_files = {k: v for k, v in folder_to_files.items() if v}
    assert len(folder_to_files) > 1

    pair_count = 90
    fused_image_folder = "fused_image"
    merged_label_path = "merged_labels.txt"
    coors_folder = "coors"
    os.makedirs(coors_folder, exist_ok=True)
    os.makedirs(fused_image_folder, exist_ok=True) 
    combinations = read_files_from_folders(train_folder)

    for pairs in combinations:
        folder1, file1, folder2, file2 = pairs
        folder1, folder2 = os.path.join(train_folder, folder1), os.path.join(train_folder, folder2)

        print(os.path.join(folder1, file1), os.path.join(folder2, file2))

        label1 = next(label for filename, label in labels if filename == folder1)
        label1_one_hot = np.array(label_to_onehot[label1])
        label2 = next(label for filename, label in labels if filename == folder2)
        label2_one_hot = np.array(label_to_onehot[label2])

        image1 = cv2.imread(os.path.join(folder1, file1))
        image2 = cv2.imread(os.path.join(folder2, file2))

        assert image1.shape == image_size
        assert image2.shape == image_size

        pair_count += 1

        fused_images, coordinates, merged_label, index = main(image1, image2, label1, label2, label1_one_hot, label2_one_hot, api_key)
        fused_image = fused_images[index]
        
        cv2.imshow("fused", fused_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        coordinate1, coordinate2 = coordinates
        os.makedirs(os.path.join(coors_folder, folder1), exist_ok=True)
        os.makedirs(os.path.join(coors_folder, folder2), exist_ok=True)
        np.save(os.path.join(coors_folder, folder1, file1[:-5]) + ".npy", coordinate1)
        np.save(os.path.join(coors_folder, folder2, file2[:-5]) + ".npy", coordinate2)

        fused_image_path = os.path.join(fused_image_folder, f"fused_{pair_count}.jpg")
        merged_label = fused_image_path + "\t" + np.array2string(merged_label, max_line_width=1000)
        cv2.imwrite(fused_image_path, fused_image)

        with open(os.path.join(fused_image_folder, merged_label_path), "a") as f:
            f.write(merged_label)
            f.write("\n")


    print("Finished")

processing(image_size, dataset_folder)