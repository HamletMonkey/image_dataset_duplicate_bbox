import os
from iou_score_vis import *
from utils.quick_img_xml import quick_plot
from utils.xml_manipulation import (
    get_xml_class_list,
    get_targeted_xml_object,
    add_xml_object,
)
import pyinputplus as pyip
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

# to view images of overlapping bounding boxes at different iou_score
def view_iouscore_img(df, IMG_PATH, XML_PATH, min_iou=0, max_iou=1):
    df_new = df[(df["iou_score"] > min_iou) & (df["iou_score"] < max_iou)]
    try:
        view_list = list(df_new["image_id"].unique())
    except:
        view_list = list(df_new.index.unique())
    print(f"Number of images: {len(view_list)}")
    quick_plot(view_list, IMG_PATH, XML_PATH)
    return view_list


def get_sub(x, duplicate_pair_list, substitute_dict):
    sub_name = substitute_dict[duplicate_pair_list.index(set(x))]
    return sub_name


def remove_duplicate_bbox(XML_PATH, threshold, OUTPUT_XML_PATH="duplicate_removed_xml"):

    """
    To remove duplicate bounding boxes on the same object.
    High iou score between the 'person' class and other objects are common especially if objects are attached/held by 'person' in images.
    Hence, if the 'person' class exist in the XML files, 2 different options will be provided for user:
        1. duplicate bbox without the class 'person'
        2. duplicate bbox including the class 'person' ONLY

    If any duplicated bounding boxes (with iou score above the threshold passed in) are not of the same object class, the user is required to enter the class name to keep.

    # Arguments
        XML_PATH: str, xml folder path
        threshold: float, ranges between 0 and 1
        OUTPUT_XML_PATH: str, default = 'duplicate_removed_xml, folder name to save the output xml files

    """

    df_iou = iou_score_plot(XML_PATH, False)
    # slice df_iou based on the input iou threshold
    df_thresh = df_iou[df_iou["iou_score"] > threshold]

    # check if object classes including person
    all_class = get_xml_class_list(XML_PATH)
    if "person" in all_class:

        question = {
            1: 'duplicate bbox without the class "person"',
            2: 'duplicate bbox including the class "person" only',
        }
        ans0 = pyip.inputMenu([question[1], question[2]], numbered=True)
        print(f"selected option: {ans0}")

        if ans0 == question[1]:
            df_target = df_thresh[
                df_thresh["bbox_class_pair"].apply(lambda x: "person" not in x)
            ]
        else:
            df_target = df_thresh[
                df_thresh["bbox_class_pair"].apply(lambda x: "person" in x)
            ]

    else:
        df_target = df_thresh

    # obtain for duplicated class pair:
    dupli_class = list()
    for item in df_target["bbox_class_pair"]:
        temp = set(item)
        if temp not in dupli_class:
            dupli_class.append(set(item))

    # get user input on merged bounding box class name:
    kw_dict = {}
    while True:
        for index, item in enumerate(dupli_class):
            if len(item) <= 1:
                # no change in class name is required
                kw_dict[index] = next(iter(item))
            if len(item) > 1:
                ans1 = pyip.inputStr(
                    prompt=f"Please enter the object-class to replace duplicated bounding boxes pair {item}:\n"
                )
                kw_dict[index] = ans1
                print(f"Your input: {ans1}\n")

        for k, v in kw_dict.items():
            print(f"Replacing {dupli_class[k]} with '{v}'")

        ans2 = pyip.inputYesNo(prompt="Confirm all inputs? Please enter: yes or no\n")

        if ans2 == "no":
            print(f"Please re-enter object-class:\n")
            continue
        else:
            print(f"Confirmation received :^)")
            break

    # create more columns for df_target
    df_target_ = df_target.copy()
    df_target_["inter_coord"] = df_target_["bbox_coord_pair"].apply(
        lambda x: iou_inter_coord(x[0], x[1])
    )
    df_target_["inter_pair"] = df_target_["bbox_class_pair"].apply(
        lambda x: get_sub(x, dupli_class, kw_dict)
    )
    df_target_ = df_target_.set_index("image_id")

    # remove duplicate bounding boxes
    to_clean = list(df_target_.index.unique())

    for item in tqdm(to_clean):
        # take into account of images with one or multiple duplicate bounding boxes:

        # ------- Step 1: get bounding boxes to drop ------- #

        checklist = []
        temp1 = df_target_.loc[item, "bbox_coord_pair"]
        # for images with only one pair of bounding boxes
        if type(temp1) == list:
            for i in temp1:
                checklist.append(i)
        # for images with more than one pair of bounding boxes
        else:
            for i in range(len(temp1)):
                for j in range(2):
                    if temp1[i][j] not in checklist:
                        checklist.append(temp1[i][j])

        # ------- Step 2: get new bounding boxes to write in ------- #

        new_coord = []
        temp2 = df_target_.loc[item, "inter_coord"]
        # for images with only one pair of bounding boxes
        if type(temp2) == list:
            new_coord.append(temp2)
        # for images with more than one pair of bounding boxes
        else:
            new_coord = list(temp2)

        # ------- Step 3: get new bounding boxes object name ------- #

        sub_class_name = []
        temp3 = df_target_.loc[item, "inter_pair"]
        # for images with only one pair of bounding boxes
        if type(temp3) == str:
            sub_class_name.append(temp3)
        # for images with more than one pair of bounding boxes
        else:
            sub_class_name = list(temp3)

        assert len(new_coord) == len(
            sub_class_name
        ), "number of coordinates and class name does not match!"

        # ------- Step 4: remove all overlapping bounding boxes ------- #

        tree = ET.parse(os.path.join(XML_PATH, f"{item}.xml"))
        root = tree.getroot()
        object_to_remove = get_targeted_xml_object(root, checklist)
        for i in object_to_remove:
            root.remove(i)

        # ------- Step 5: write in new bounding boxes ------- #

        final_new_coord = []
        final_sub_class_name = []
        for index, obj in enumerate(new_coord):
            if obj not in final_sub_class_name:
                # take the name as:
                name = sub_class_name[index]
                # then write in new object in XML tree
                add_xml_object(root, name, obj)
                final_new_coord.append(obj)
                final_sub_class_name.append(name)

        assert len(final_new_coord) == len(
            final_sub_class_name
        ), "number of FINAL coordinates and class name does not match!"

        # ------- Step 6: save the new XML file ------- #

        ET.indent(tree, space="\t", level=0)
        # ET.dump(tree)
        save_xml_folder = OUTPUT_XML_PATH
        os.makedirs(save_xml_folder, exist_ok=True)
        tree.write(os.path.join(save_xml_folder, f"{item}.xml"))

    return


def range_limited_float_type(arg):
    """Type function for argparse - a float within some predefined bounds"""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    MIN_VAL, MAX_VAL = 0, 1
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError(
            "Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL)
        )
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xmlpath",
        type=str,
        required=True,
        help="path to XML annotations folder",
    )
    parser.add_argument(
        "-t",
        type=range_limited_float_type,
        required=True,
        help="iou score threshold, float range from 0 to 1",
    )
    parser.add_argument(
        "--n_xmlpath",
        type=str,
        default="duplicate_removed_xml",
        help="path to save output XML annotations",
    )

    args = parser.parse_args()

    remove_duplicate_bbox(
        XML_PATH=args.xmlpath,
        threshold=args.t,
        OUTPUT_XML_PATH=args.n_xmlpath,
    )
