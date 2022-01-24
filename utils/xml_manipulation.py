"""
Different functions for XML manipulations!
"""


def get_xml_bbox(root):
    result = []
    for object in root.findall("object"):
        xmin = int(object.find("bndbox/xmin").text)
        ymin = int(object.find("bndbox/ymin").text)
        xmax = int(object.find("bndbox/xmax").text)
        ymax = int(object.find("bndbox/ymax").text)
        result.append([xmin, ymin, xmax, ymax])
    return result


def get_xml_bbox_area(root):
    areas = []
    for object in root.findall("object"):
        xmin = int(object.find("bndbox/xmin").text)
        ymin = int(object.find("bndbox/ymin").text)
        xmax = int(object.find("bndbox/xmax").text)
        ymax = int(object.find("bndbox/ymax").text)
        area = int((ymax - ymin) * (xmax - xmin))
        areas.append(area)
    return areas


def get_xml_width(root):
    width = int(root.find("size/width").text)
    return width


def get_xml_height(root):
    height = int(root.find("size/height").text)
    return height


def get_targeted_xml_object(root, wanted_bbox_list):
    """
    return list of element tree object if the any bounding box is in the passed in list
    """
    target_object = []
    for object in root.findall("object"):
        xmin = int(object.find("bndbox/xmin").text)
        ymin = int(object.find("bndbox/ymin").text)
        xmax = int(object.find("bndbox/xmax").text)
        ymax = int(object.find("bndbox/ymax").text)
        if [xmin, ymin, xmax, ymax] in wanted_bbox_list:
            target_object.append(object)
    return target_object


def get_xml_class_list(XML_PATH):

    # import library
    import os
    from tqdm import tqdm
    import xml.etree.ElementTree as ET
    from tools import get_filelist

    xml_files = get_filelist(XML_PATH)
    class_list = []
    for item in tqdm(xml_files):
        tree = ET.parse(os.path.join(XML_PATH, f"{item}.xml"))
        root = tree.getroot()
        for object in root.findall("object"):
            name = object.find("name").text
            if name not in class_list:
                class_list.append(name)
    return class_list


def add_xml_object(root, name, bbox_coordinates):

    # import library
    import xml.etree.ElementTree as ET

    new_object = ET.SubElement(root, "object")
    new_name = ET.SubElement(new_object, "name")
    new_name.text = name
    new_pose = ET.SubElement(new_object, "pose")
    new_pose.text = "Unspecified"
    new_trun = ET.SubElement(new_object, "truncated")
    new_trun.text = "0"
    new_diff = ET.SubElement(new_object, "difficult")
    new_diff.text = "0"
    new_bbox = ET.SubElement(new_object, "bndbox")
    new_xmin = ET.SubElement(new_bbox, "xmin")
    new_xmin.text = str(bbox_coordinates[0])
    new_ymin = ET.SubElement(new_bbox, "ymin")
    new_ymin.text = str(bbox_coordinates[1])
    new_xmax = ET.SubElement(new_bbox, "xmax")
    new_xmax.text = str(bbox_coordinates[2])
    new_ymax = ET.SubElement(new_bbox, "ymax")
    new_ymax.text = str(bbox_coordinates[3])

    return


def drop_xml_small_bbox(root, max_area=400, list_to_keep=None):
    """
    This function loops through all bounding boxes in the XML annotation file passed in.
    If the 2 conditions passed in are met, the bounding box will be dropped.

    Arguments:
        root: xml.Elementree object, root element of the parsed xml tree
        max_area: int, bounding box area smaller than this threshold will be removed
        list_to_keep: list, a list of lists contains bounding boxes coordinates [xmin, ymin, xmax, ymax] to be kept

    Returns:
        remain_box: list, a list of lists contains remaining bounding boxes

    """
    remain_bbox = []  # store bboxes kept
    for object in root.findall("object"):
        xmin = int(object.find("bndbox/xmin").text)
        ymin = int(object.find("bndbox/ymin").text)
        xmax = int(object.find("bndbox/xmax").text)
        ymax = int(object.find("bndbox/ymax").text)
        area = int((ymax - ymin) * (xmax - xmin))

        ## 2 conditions:
        # condition1:
        cond1 = area <= max_area
        # condition2:
        cond2 = [xmin, ymin, xmax, ymax] not in list_to_keep

        try:
            if max_area is None and list_to_keep is None:
                print("Invalid input: both conditions cannot be NoneType")
            elif cond1 and list_to_keep is None:
                root.remove(object)
            elif cond2 and max_area is None:
                root.remove(object)
            elif cond1 and cond2:
                root.remove(object)
            else:
                remain_bbox.append([xmin, ymin, xmax, ymax])
        except TypeError:
            # exclude cases of either condition arguments is NoneType
            remain_bbox.append([xmin, ymin, xmax, ymax])
    return remain_bbox


def update_crop_xml_bbox(root, w, h, x_shift, y_shift):
    """
    Return new bounding boxes value of cropped image

    Arguments:
        root: xml.Elementree object, root element of the parsed xml tree
        w: int, new image width after cropped
        h: int, new image height after cropped
        x_shift: int, number of pixel shifted from original image, in x-direction
        y_shift: int, number of pixel shifted from original image, in y-direction

    """
    root.find("size/width").text = str(w)
    root.find("size/height").text = str(h)
    for object in root.findall("object"):
        # for shifted xmin
        xmin = int(object.find("bndbox/xmin").text)
        object.find("bndbox/xmin").text = str(xmin - x_shift)
        # for shifted ymin
        ymin = int(object.find("bndbox/ymin").text)
        object.find("bndbox/ymin").text = str(ymin - y_shift)
        # for shifted xmax
        xmax = int(object.find("bndbox/xmax").text)
        object.find("bndbox/xmax").text = str(xmax - x_shift)
        # for shited ymax
        ymax = int(object.find("bndbox/ymax").text)
        object.find("bndbox/ymax").text = str(ymax - y_shift)
    return
