import os, sys
import numpy as np 
import nibabel as nib
import json
from distutils.util import strtobool
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

LABEL_DICT = "label_mappings/label_dict_127_abdomenAtlas3-1.json"   # modified for abdomenAtlas3-1

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

        
# def seperate_class(data):

#     def save_each_class(vol_name, pred_nii, class_list, label_list, label_prompt):
#         pred_data = pred_nii.get_fdata().astype(np.uint8)
#         for cls_id in label_prompt:
#             if cls_id == 28:    # lung_left
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==28) + (pred_data==29)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join("./eval", vol_name, "predictions", "lung_left.nii.gz")
#                     )
#             if cls_id == 30:    # lung_right
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==30) + (pred_data==31) + (pred_data==32)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join("./eval", vol_name, "predictions", "lung_right.nii.gz")
#                     )
#             class_nii = nib.nifti1.Nifti1Image((pred_data==cls_id).astype(np.uint8), pred_nii.affine)
#             class_nii.set_qform(pred_nii.get_qform())
#             class_nii.set_sform(pred_nii.get_sform())
#             class_nii.to_filename(
#                 os.path.join("./eval", vol_name, "predictions", f"{class_list[label_list.index(cls_id)].replace(' ', '_')}.nii.gz")
#                 )
#     # Access the filename of the saved image    
#     filename = data['image_meta_dict']['filename_or_obj']    
#     volume_name = filename.split("/")[-2]

#     label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT - SEVEN_CLS)   # 117

#     with open(LABEL_DICT, "r") as f:
#         label_dict = json.load(f)
#     class_list = list(label_dict.keys())
#     label_list = list(label_dict.values())

#     if not os.path.exists(f"./eval/{volume_name}/predictions"):
#         os.mkdir(f"./eval/{volume_name}/predictions")

#     pred_nii = nib.load(
#         os.path.join("./eval", volume_name, "ct_step1_117.nii.gz")
#     ) 
#     save_each_class(volume_name, pred_nii, class_list, label_list, label_prompt)

#     os.remove(os.path.join("./eval", volume_name, "ct_step1_117.nii.gz"))

#     return data

def build_input_list(input_dir, input_suffix, output_dir):
    def rprint(*string):
        print("\033[31m", *string, "\033[0m")

    print("build input list...")

    # List all files in the directory (os.scandir is super fast)
    filtered_files = [os.path.join(entry.path, input_suffix) for entry in os.scandir(input_dir)]
    # Sort the filtered file paths
    input_list_path = sorted(filtered_files)

    input_dict = {x.split("/")[-2]:x for x in input_list_path} # if 32584 <= int(x.split("/")[-2][-5:]) and int(x.split("/")[-2][-5:]) <= 34427
    rprint("[INFO]", "[Total Volumes Detected]", len(input_dict))

    eval_list_path = glob.glob(os.path.join(output_dir, "*", "predictions"))
    eval_list_volume = list(map(lambda x: x.split("/")[-2], eval_list_path))    # BDMAP_XXXXXXXX
    eval_status_list = list(map(lambda x: len(glob.glob(os.path.join(x, "*.nii.gz"))) == 125, eval_list_path))
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
        rprint("\033[31mAll volumes have already been inferenced and stored in `./eval/`. Enjoy.\033[0m")
        sys.exit(0)
    return input_list


def build_input_list_from_csv(input_dir, input_csv, output_dir):
    def rprint(*string):
        print("\033[31m", *string, "\033[0m")

    print("build input list... (from csv)")

    # building list using csv
    df = pd.read_csv(input_csv)
    # >>>>>> The following part might differ due to different csv file >>>>>>
    patient_id_list = sorted(df["Target ID"].tolist())  # [BDMAP_XXXXXXXX, ]
    input_list_path = list(map(lambda x:os.path.join(input_dir, x, "ct.nii.gz"), patient_id_list))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    input_dict = {x.split("/")[-2]:x for x in input_list_path} # if 32584 <= int(x.split("/")[-2][-5:]) and int(x.split("/")[-2][-5:]) <= 34427
    rprint("[INFO]", "[Total Volumes Detected]", len(input_dict))

    eval_list_path = glob.glob(os.path.join(output_dir, "*", "predictions"))
    eval_list_volume = list(map(lambda x: x.split("/")[-2], eval_list_path))    # BDMAP_XXXXXXXX
    eval_status_list = list(map(lambda x: len(glob.glob(os.path.join(x, "*.nii.gz"))) == 125, eval_list_path))
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
    return input_list



# def seperate_class2(pred_dir):

#     def save_each_class(vol_name, pred_nii, class_list, label_list, label_prompt, output_root):
#         pred_data = pred_nii.get_fdata().astype(np.uint8)
#         for cls_id in label_prompt:

#             class_nii = nib.nifti1.Nifti1Image((pred_data==cls_id).astype(np.uint8), pred_nii.affine)
#             class_nii.set_qform(pred_nii.get_qform())
#             class_nii.set_sform(pred_nii.get_sform())
#             class_nii.to_filename(
#                 os.path.join(output_root, "eval", vol_name, "predictions", f"{class_list[label_list.index(cls_id)].replace(' ', '_')}.nii.gz")
#                 )
            
#             if cls_id == 1: # liver (merge with hepatic vessel)
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==1) + (pred_data==25)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join(output_root, "eval", vol_name, "predictions", "liver.nii.gz")
#                     )
                
#             if cls_id == 28:    # lung_left
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==28) + (pred_data==29)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join(output_root, "eval", vol_name, "predictions", "lung_left.nii.gz")
#                     )
#             if cls_id == 30:    # lung_right
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==30) + (pred_data==31) + (pred_data==32)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join(output_root, "eval", vol_name, "predictions", "lung_right.nii.gz")
#                     )
                
#             if cls_id == 20:    # lung (all lobes)
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==28) + (pred_data==29) + (pred_data==30) + (pred_data==31) + (pred_data==32)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join(output_root, "eval", vol_name, "predictions", "lung.nii.gz")
#                     )
#             if cls_id == 2:    # kidney (left and right)
#                 class_nii = nib.nifti1.Nifti1Image(((pred_data==5) + (pred_data==14)).astype(np.uint8), pred_nii.affine)
#                 class_nii.set_qform(pred_nii.get_qform())
#                 class_nii.set_sform(pred_nii.get_sform())
#                 class_nii.to_filename(
#                     os.path.join(output_root, "eval", vol_name, "predictions", "kidney.nii.gz")
#                     )
#     # Access the filename of the saved image    
#     output_root = "/ccvl/net/ccvl15/tlin67/CCVL/VISTA3D"
#     volume_name = pred_dir.split("/")[-1]

#     label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT)   # 117

#     with open(LABEL_DICT, "r") as f:
#         label_dict = json.load(f)
#     class_list = list(label_dict.keys())
#     label_list = list(label_dict.values())

#     if not os.path.exists(f"{output_root}/eval/{volume_name}/predictions"):
#         os.makedirs(f"{output_root}/eval/{volume_name}/predictions")
#     else:
#         if len(glob.glob(os.path.join(output_root, "eval", volume_name, "predictions", "*"))) == 129:
#             return
#         else:
#             pass

#     pred_nii = nib.load(
#         os.path.join(pred_dir, "vista3d_allSegments.nii.gz")
#     ) 

#     save_each_class(volume_name, pred_nii, class_list, label_list, label_prompt, output_root)


#     # return data


def seperate_class(data):

    def save_each_class(vol_name, pred_nii, class_list, label_list, label_prompt, output_root):
        pred_data = pred_nii.get_fdata().astype(np.uint8)
        for cls_id in label_prompt:
            # DON'T SAVE THESE:
            if cls_id == 116 or cls_id == 117:
                if cls_id == 116:   # kidney cyst (left, merged with right kindney cyst)
                    class_nii = nib.nifti1.Nifti1Image(((pred_data==116) + (pred_data==117)).astype(np.uint8), pred_nii.affine)
                    class_nii.set_qform(pred_nii.get_qform())
                    class_nii.set_sform(pred_nii.get_sform())
                    class_nii.to_filename(
                        os.path.join(output_root, vol_name, "predictions", "_kidney_cyst.nii.gz")
                        )
                continue
            
            # NORMAL CASES
            class_nii = nib.nifti1.Nifti1Image((pred_data==cls_id).astype(np.uint8), pred_nii.affine)
            class_nii.set_qform(pred_nii.get_qform())
            class_nii.set_sform(pred_nii.get_sform())
            class_nii.to_filename(
                os.path.join(output_root, vol_name, "predictions", f"{class_list[label_list.index(cls_id)].replace(' ', '_')}.nii.gz")
                )
            
            # SPECIAL CASES:
            if cls_id == 1: # liver (merge with hepatic vessel)
                class_nii = nib.nifti1.Nifti1Image(((pred_data==1) + (pred_data==25)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join(output_root, vol_name, "predictions", "liver.nii.gz")
                    )
            if cls_id == 28:    # lung_left
                class_nii = nib.nifti1.Nifti1Image(((pred_data==28) + (pred_data==29)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join(output_root, vol_name, "predictions", "lung_left.nii.gz")
                    )
            if cls_id == 30:    # lung_right
                class_nii = nib.nifti1.Nifti1Image(((pred_data==30) + (pred_data==31) + (pred_data==32)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join(output_root, vol_name, "predictions", "lung_right.nii.gz")
                    )
                
    # Access the filename of the saved image    
    # NOTE: MODITY THIS TO ONE-HOT OUTPUT PATH!
    output_root = os.environ["VISTA3D_OUTPUT_DIR"]  # brillant my friend!

    filename = data['image_meta_dict']['filename_or_obj']    
    volume_name = filename.split("/")[-2]

    label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT)   # 117

    with open(LABEL_DICT, "r") as f:
        label_dict = json.load(f)
    class_list = list(label_dict.keys())
    label_list = list(label_dict.values())

    if not os.path.exists(f"{output_root}/{volume_name}/predictions"):
        os.makedirs(f"{output_root}/{volume_name}/predictions")

    pred_nii = nib.load(
        os.path.join(output_root, volume_name, "ct_step1_117.nii.gz")
    ) 

    save_each_class(volume_name, pred_nii, class_list, label_list, label_prompt, output_root)

    os.remove(os.path.join(output_root, volume_name, "ct_step1_117.nii.gz"))

    return data


def seperate_class_pathInput(pred_path):   # `seperate_class`, but use prediction path as input

    def save_each_class(vol_name, pred_nii, class_list, label_list, label_prompt, output_root):
        pred_data = pred_nii.get_fdata().astype(np.uint8)
        for cls_id in label_prompt:
            # DON'T SAVE THESE:
            if cls_id == 116 or cls_id == 117:
                if cls_id == 116:   # kidney cyst (left, merged with right kindney cyst)
                    class_nii = nib.nifti1.Nifti1Image(((pred_data==116) + (pred_data==117)).astype(np.uint8), pred_nii.affine)
                    class_nii.set_qform(pred_nii.get_qform())
                    class_nii.set_sform(pred_nii.get_sform())
                    class_nii.to_filename(
                        os.path.join(output_root, vol_name, "predictions", "_kidney_cyst.nii.gz")
                        )
                continue
            
            # NORMAL CASES
            class_nii = nib.nifti1.Nifti1Image((pred_data==cls_id).astype(np.uint8), pred_nii.affine)
            class_nii.set_qform(pred_nii.get_qform())
            class_nii.set_sform(pred_nii.get_sform())
            class_nii.to_filename(
                os.path.join(output_root, vol_name, "predictions", f"{class_list[label_list.index(cls_id)].replace(' ', '_')}.nii.gz")
                )
            
            # SPECIAL CASES:
            if cls_id == 1: # liver (merge with hepatic vessel)
                class_nii = nib.nifti1.Nifti1Image(((pred_data==1) + (pred_data==25)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join(output_root, vol_name, "predictions", "liver.nii.gz")
                    )
            if cls_id == 28:    # lung_left
                class_nii = nib.nifti1.Nifti1Image(((pred_data==28) + (pred_data==29)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join(output_root, vol_name, "predictions", "lung_left.nii.gz")
                    )
            if cls_id == 30:    # lung_right
                class_nii = nib.nifti1.Nifti1Image(((pred_data==30) + (pred_data==31) + (pred_data==32)).astype(np.uint8), pred_nii.affine)
                class_nii.set_qform(pred_nii.get_qform())
                class_nii.set_sform(pred_nii.get_sform())
                class_nii.to_filename(
                    os.path.join(output_root, vol_name, "predictions", "lung_right.nii.gz")
                    )
                
    filename = pred_path#data['image_meta_dict']['filename_or_obj']    
    volume_name = filename.split("/")[-2]

    label_prompt = list(set([i + 1 for i in range(132)]) - IGNORE_PROMPT)   # 117

    with open(LABEL_DICT, "r") as f:
        label_dict = json.load(f)
    class_list = list(label_dict.keys())
    label_list = list(label_dict.values())

    if not os.path.exists(f"{output_root}/{volume_name}/predictions"):
        os.makedirs(f"{output_root}/{volume_name}/predictions")

    pred_nii = nib.load(
        os.path.join(pred_path)
    ) 

    save_each_class(volume_name, pred_nii, class_list, label_list, label_prompt, output_root)



if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="Parser for pred_root and output_root")
        parser.add_argument("--pred_root", type=str, required=True, help="Path to the prediction root directory"\
                            "e.g. `results/vista3d_allSegments_20k`")
        parser.add_argument("--output_root", type=str, required=True, help="Path to the output root directory"\
                            "e.g. `eval_allSegments_20k`")
        args = parser.parse_args()
        return args
    
    args = parse_args()
    output_root = args.output_root  # to pass to `seperate_class_pathInput`

    # Use multiprocessing Pool to process cases in parallel
    num_processes = cpu_count()

    # case_folders = glob.glob(os.path.join(pred_root, "*"))
    case_folders = build_input_list(args.pred_root, "*.nii.gz", args.output_root)

    # for pred_dir in tqdm(case_folders):
    #     seperate_class3_pathInput(pred_dir)
    #     break

    with Pool(num_processes) as pool:
        # Use tqdm to display progress bar
        results = []
        for res in tqdm(pool.imap_unordered(seperate_class_pathInput, case_folders), total=len(case_folders), desc='Processing cases'):
            if res:
                print(res, flush=True)
                results.append(res)

