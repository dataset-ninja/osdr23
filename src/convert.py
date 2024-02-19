import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
from glob import glob
import imagesize

from tqdm import tqdm


def create_ann(img_path):
    width, height = imagesize.get(img_path)
    anns = 1
    path_parts = img_path.split(os.path.sep)
    subds_name = path_parts[-2]
    ds_name = path_parts[-3]
    ds_tags = meta_dict[ds_name].get(subds_name)

    img_tags = [sly.Tag(tag_metas.get(tag_name), tag_value) for tag_name, tag_value in ds_tags]
    labels = []
    sly.Label()
    return sly.Annotation((height, width), labels, img_tags)


def update_meta_dict(path):
    path_to_txt_file = glob(os.path.join(path, "*.txt"))
    subdict_key = ""
    subdict_values = []
    keywords_to_skip = ["CAMERA", "LIDAR", "RADAR"]
    with open(path_to_txt_file) as file:
        for line in file.readlines()[7:]:
            if any(keyword in line for keyword in keywords_to_skip):
                continue
            if "#" in line:
                subdict = dict(zip(subdict_key, subdict_values))
                meta_dict[os.path.basename(path)] = subdict
                subdict_values.clear()
                subdict_key = ""
            line_parts = line.strip().split(":")
            if line_parts[0] == "data_folder":
                subdict_key = line_parts[1]
            else:
                subdict_values.append((line_parts[0], line_parts[1]))


meta_dict = {}
tag_metas = []  # todo
obj_classes = []  # todo


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    dataset_path = "/mnt/c/users/german/documents/osdr"

    project = api.project.create(workspace_id, project_name)
    meta = sly.ProjectMeta(obj_classes.values(), tag_metas)

    foldernames_to_skip = ["lidar", "readme_img"]

    datasets_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    pbar = tqdm(total=len(datasets_paths), desc=f"Processing datasets...")
    for ds_path in datasets_paths:
        ds_name = os.path.basename(ds_path)
        if ds_name in foldernames_to_skip:
            continue
        update_meta_dict(ds_path)

        dataset = api.dataset.create(project.id, ds_name)

        image_paths = [
            (path.split(os.path.sep)[-2] + "_" + os.path.basename(path), path)
            for path in glob(os.path.join(ds_path, "*", "*.png"))
            if path.split(os.path.sep)[-2] not in foldernames_to_skip
        ]
        for img_names_batch, img_paths_batch in sly.batched(image_paths):
            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_paths_batch)
            img_ids = [img_info.id for img_info in img_infos]
            # tag_values = (subfolder_name, captions.get(subfolder_name))
            anns = [create_ann(img_path) for img_path in img_paths_batch]
            api.annotation.upload_anns(img_ids, anns)
        pbar.update(1)
    pbar.close()

    # return project
