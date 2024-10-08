import os, sys
import numpy as np 
import nibabel as nib
import json
from distutils.util import strtobool
import glob
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
            if cls_id == 28:    # lung_left
                class_nii = nib.nifti1.Nifti1Image(((pred_data==28) + (pred_data==29)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join("./eval", vol_name, "predictions", "lung_left.nii.gz")
                    )
            if cls_id == 30:    # lung_right
                class_nii = nib.nifti1.Nifti1Image(((pred_data==30) + (pred_data==31) + (pred_data==32)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join("./eval", vol_name, "predictions", "lung_right.nii.gz")
                    )
            class_nii = nib.nifti1.Nifti1Image((pred_data==cls_id).astype(np.uint8), pred_nii.affine)
            class_nii.set_qform(pred_nii.get_qform())
            class_nii.set_sform(pred_nii.get_sform())
            class_nii.to_filename(
                os.path.join("./eval", vol_name, "predictions", f"{class_list[label_list.index(cls_id)].replace(' ', '_')}.nii.gz")
                )
    # Access the filename of the saved image    
    filename = data['image_meta_dict']['filename_or_obj']    
    volume_name = filename.split("/")[-2]

    label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT - SEVEN_CLS)   # 117

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

    return data

def build_input_list(input_dir, input_suffix, output_dir):
    def rprint(*string):
        print("\033[31m", *string, "\033[0m")

    input_list_path = sorted(glob.glob(os.path.join(input_dir, "*", input_suffix)))
    input_dict = {x.split("/")[-2]:x for x in input_list_path}
    rprint("[INFO]", "[Total Volumes Detected]", len(input_dict))

    eval_list_path = glob.glob(os.path.join(output_dir, "*", "predictions"))
    eval_list_volume = list(map(lambda x: x.split("/")[-2], eval_list_path))    # BDMAP_XXXXXXXX
    eval_status_list = list(map(lambda x: len(glob.glob(os.path.join(x, "*.nii.gz"))) == 117 + 2, eval_list_path))
    eval_completed_list = np.asarray(eval_list_volume)[eval_status_list]

    already_completed_list = [(volume_name if volume_name in eval_completed_list else None) \
                              for volume_name in input_dict.keys()]
    already_completed_list = list(filter(lambda x: x is not None, already_completed_list))
    for key in already_completed_list: # get remaining volumes by deleting completed volumes
        input_dict.pop(key)

    rprint("[INFO]", "[Already Inferenced]", len(already_completed_list))

    input_list = list(input_dict.values())
    rprint("[INFO]", "[Remaining Volumes]", len(input_list))

    if len(input_list) == 0:
        raise ValueError("\033[31mAll volumes have already been inferenced and stored in `./eval/`. Enjoy.\033[0m")
    sys.exit(0)
    return input_list


if __name__ == "__main__":
    folder = sys.argv[1]
    second_stage = bool(strtobool(sys.argv[2]))

    label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT - SEVEN_CLS)   # 117

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