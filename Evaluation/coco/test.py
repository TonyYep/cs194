import json

# COCO 数据集的下载链接
# urls = [
#     "http://images.cocodataset.org/zips/train2017.zip",
#     "http://images.cocodataset.org/zips/val2017.zip",
#     "http://images.cocodataset.org/zips/test2017.zip",
#     "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
# ]

# # 下载并保存文件的目录
# save_dir = "./coco_dataset"
# os.makedirs(save_dir, exist_ok=True)


# # 使用wget下载文件
# for url in urls:
#     filename = os.path.join(save_dir, url.split("/")[-1])
# #     wget.download(url, filename)
# #     print(f"Downloaded: {filename}")
# def create_coco_subset(annotation_file, output_file, subset_percentage=0.1):
#     """
#     从 COCO 数据集的标注文件中创建一个子集，按指定比例随机选择。

#     参数：
#     - annotation_file: str，原始的 COCO 标注文件路径
#     - output_file: str，输出的子集标注文件路径
#     - subset_percentage: float，选择子集的比例，默认 0.1（即 10%）
#     """
#     # 加载 COCO 标注文件
#     with open(annotation_file, "r") as f:
#         coco_data = json.load(f)

#     # 获取所有图像的ID
#     image_ids = [image["id"] for image in coco_data["images"]]

#     # 随机选择子集
#     subset_size = int(len(image_ids) * subset_percentage)
#     selected_image_ids = random.sample(image_ids, subset_size)

#     # 创建新的子集数据
#     new_images = []
#     new_annotations = []
#     selected_image_ids_set = set(selected_image_ids)

#     # 遍历所有图像，保留所选图像
#     for image in coco_data["images"]:
#         if image["id"] in selected_image_ids_set:
#             new_images.append(image)

#     # 遍历所有标注，保留与所选图像ID匹配的标注
#     for annotation in coco_data["annotations"]:
#         if annotation["image_id"] in selected_image_ids_set:
#             new_annotations.append(annotation)

#     # 创建子集的 COCO 格式 JSON 数据
#     coco_subset = {
#         "images": new_images,
#         "annotations": new_annotations,
#         "categories": coco_data["categories"],
#     }

#     # 保存子集标注文件
#     with open(output_file, "w") as f:
#         json.dump(coco_subset, f)


# create_coco_subset(
#     "annotations/instances_train2017.json", "annotations/sub_instances_train2017.json"
# )
# 加载 COCO 的 annotations 文件
# with open("dataset/instances_val2017.json", "r") as f:
#     coco_data = json.load(f)

# # 遍历每个实例并打印 category_id
# for annotation in coco_data["annotations"]:
#     category_id = annotation["category_id"]
#     print(category_id)
# with open("dataset/sub_instances_train2017.json", "r") as f:
#     coco_data = json.load(f)

# # 获取类别 ID 和名称的映射
# category_map = {
#     category["id"]: category["name"] for category in coco_data["categories"]
# }

# # 创建一个新的连续类别 ID 映射，将原始 ID 映射到 1-80 范围内
# sorted_category_ids = sorted(category_map.keys())  # 按照原始 ID 排序
# category_id_mapping = {
#     original_id: i + 1 for i, original_id in enumerate(sorted_category_ids)
# }  # 映射到 1-80
# with open("category_id_mapping.json", "w") as f:
#     json.dump(category_id_mapping, f)
# # 更新 annotations 中的 category_id
# for annotation in coco_data["annotations"]:
#     original_category_id = annotation["category_id"]
#     new_category_id = category_id_mapping.get(
#         original_category_id, -1
#     )  # 如果找不到对应 ID，返回 -1（表示未知）
#     annotation["category_id"] = new_category_id

# # 保存修改后的数据（可选）
# with open("dataset/sub_instances_train2017.json", "w") as f:
#     json.dump(coco_data, f)

# # 打印前几个更新的 annotation 来确认
# for annotation in coco_data["annotations"][:5]:
#     print(annotation)
# 读取之前保存的 category_id_mapping
with open("category_id_mapping.json", "r") as f:
    category_id_mapping = json.load(f)

# 加载 COCO 的 val 数据集注释文件
with open("annotations/instances_val2017.json", "r") as f:
    coco_val_data = json.load(f)

# 更新 val 数据集的 annotations 中的 category_id
for annotation in coco_val_data["annotations"]:
    original_category_id = annotation["category_id"]
    new_category_id = category_id_mapping.get(
        str(original_category_id), -1
    )  # 如果找不到对应 ID，返回 -1（表示未知）
    annotation["category_id"] = new_category_id

# 保存更新后的 val 数据集（可选）
with open("dataset/instances_val2017.json", "w") as f:
    json.dump(coco_val_data, f)

# 打印前几个更新的 annotation 来确认
for annotation in coco_val_data["annotations"][:5]:
    print(annotation)
