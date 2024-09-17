import os, sys
import numpy as np 
import nibabel as nib
import json
from distutils.util import strtobool
# from tqdm import tqdm

LABEL_DICT = "label_dict_127_touchstone.json"   # modified for touchstone label name

IGNORE_PROMPT = set(
    [   
        # delete for overlapping:
        2,  # kidney
        20,  # lung
        21,  # bone

        # delete for deprecated:    
        16,  # prostate or uterus
        18,  # rectum
        129,    # kidney mass
        130,    # liver tumor
        131,    # vertebrae L6
    ]
)  

SEVEN_CLS = set([23,24,25,26,27,128,132])   # 2nd stage inference

        
def seperate_class(data):

    def save_each_class(vol_name, pred_nii, class_list, label_list, label_prompt):
        pred_data = pred_nii.get_fdata().astype(np.uint8)
        for cls_id in label_prompt:
            class_nii = nib.nifti1.Nifti1Image((pred_data==cls_id).astype(np.uint8), pred_nii.affine)
            class_nii.set_qform(pred_nii.get_qform())
            class_nii.set_sform(pred_nii.get_sform())
            class_nii.to_filename(
                os.path.join("./eval", vol_name, "predictions", f"{class_list[label_list.index(cls_id)].replace(' ', '_')}.nii.gz")
                )
    # Access the filename of the saved image    
    # print(f"\033[31m{data.keys()}\033[0m")
    # print(f"{[(k, v) for (k, v) in data.items()]}")
    filename = data['image_meta_dict']['filename_or_obj']    
    volume_name = filename.split("/")[-2]

    label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT)   # 117

    with open(LABEL_DICT, "r") as f:
        label_dict = json.load(f)
    class_list = list(label_dict.keys())
    label_list = list(label_dict.values())

    if not os.path.exists(f"./eval/{volume_name}/predictions"):
        os.mkdir(f"./eval/{volume_name}/predictions")

    pred_nii = nib.load(
        os.path.join("./eval", volume_name, "ct_step1_117.nii.gz")
    ) 
    save_each_class(volume_name, pred_nii, class_list, label_list, label_prompt)

    os.remove(os.path.join("./eval", volume_name, "ct_step1_117.nii.gz"))

    """TODO
    1. align classes with Touchstone
    2. add logger / resume feature
    """

    return data


if __name__ == "__main__":
    folder = sys.argv[1]
    second_stage = bool(strtobool(sys.argv[2]))

    label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT)   # 117

    with open(LABEL_DICT, "r") as f:
        label_dict = json.load(f)
    class_list = list(label_dict.keys())
    label_list = list(label_dict.values())

    if not os.path.exists(f"./eval/{folder}/predictions"):
        os.mkdir(f"./eval/{folder}/predictions")

    pred_nii = nib.load(
        os.path.join("./eval", folder, "ct_step1_117.nii.gz")
    )
    save_each_class(folder, pred_nii, class_list, label_list, label_prompt)

    if second_stage:
        pred_nii = nib.load(
            os.path.join("./eval", folder, "ct_step2_7.nii.gz")
        )
        save_each_class(folder, pred_nii, class_list, label_list, list(SEVEN_CLS))