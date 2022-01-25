import os
from iou_score_vis import *
from utils.xml_manipulation import get_targeted_xml_object
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse


def remove_duplicate_pseudo_bbox(
    XML_PATH,
    threshold,
    OUTPUT_XML_PATH="duplicate_removed_pseudo_xml",
):

    """
    To remove duplicating bounding boxes on the same object for annotations generated from pseudo labelling.
    This function will keep the bounding box with the highest confidence score and remove all others.

    # Arguments
        XML_PATH: str, xml folder path
        threshold: float, ranges between 0 and 1
        OUTPUT_XML_PATH: str, default = 'duplicate_removed_xml, folder name to save the output xml files

    """

    df_iou = iou_score_plot(XML_PATH, False)
    # slice df_iou based on the input iou threshold
    df_thresh = df_iou[df_iou["iou_score"] > threshold]
    df_thresh = df_thresh.set_index("image_id")

    to_clean = list(df_thresh.index.unique())

    for item in tqdm(to_clean):
        # ------- Step 1: get bounding boxes to drop ------- #

        checklist = []
        temp = df_thresh.loc[item, "bbox_coord_pair"]
        # for images with only one pair of bounding box
        if type(temp) == list:
            for i in temp:
                checklist.append(i)
        # for images with more than one pair of bounding box
        else:
            for i in range(len(temp)):
                for j in range(2):
                    if temp[i][j] not in checklist:
                        checklist.append(temp[i][j])

        # ------- Step 2: get the highest confidence score ------- #

        tree = ET.parse(os.path.join(XML_PATH, f"{item}.xml"))
        root = tree.getroot()
        overlapping_boxes = []
        wanted_object = get_targeted_xml_object(root, checklist)
        for object in wanted_object:
            name = object.find("name").text
            confidence_score = object.find("bndbox/confidence").text
            xmin = int(object.find("bndbox/xmin").text)
            ymin = int(object.find("bndbox/ymin").text)
            xmax = int(object.find("bndbox/xmax").text)
            ymax = int(object.find("bndbox/ymax").text)
            overlapping_boxes.append([name, confidence_score, xmin, ymin, xmax, ymax])

        overlapping_boxes = sorted(overlapping_boxes, key=lambda x: x[1], reverse=True)
        bboxes_after_nms = []

        # to consider the part where multiple items: IoU more than 0.75

        iou_threshold = 0.8

        while overlapping_boxes:
            chosen_box = overlapping_boxes.pop(0)
            overlapping_boxes = [
                box
                for box in overlapping_boxes
                if (score_iou(chosen_box[2:], box[2:]) < iou_threshold)
            ]
            bboxes_after_nms.append(chosen_box)

        # ------- Step 3: remove unwanted objects (keep only highest confidence score) ------- #

        for object in wanted_object:
            name = object.find("name").text
            confidence_score = object.find("bndbox/confidence").text
            xmin = int(object.find("bndbox/xmin").text)
            ymin = int(object.find("bndbox/ymin").text)
            xmax = int(object.find("bndbox/xmax").text)
            ymax = int(object.find("bndbox/ymax").text)
            result = [name, confidence_score, xmin, ymin, xmax, ymax]
            if result not in bboxes_after_nms:
                root.remove(object)

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
        default="duplicate_removed_pseudo_xml",
        help="path to save output XML annotations",
    )

    args = parser.parse_args()

    remove_duplicate_pseudo_bbox(
        XML_PATH=args.xmlpath,
        threshold=args.t,
        OUTPUT_XML_PATH=args.n_xmlpath,
    )
