import os
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt


def quick_plot(filename_list, IMG_PATH, XML_PATH):
    """
    Provides a quick visualisation of image with bounding box(es), based on list of image filename passed in.
    25 images are chosen randomly for visualisation, unless the filename_list pass in contains lesser than 25 images.
    This is useful to view specific images with their bounding boxes.

    # Arguments
        filename_list: list, list of image file names
        IMG_PATH: path, image folder path
        ANN_PATH: path, annotation folder path

    # Returns
        A visualization plot (with maximum 25 subplots) of the image with bounding box(es)

    """
    if len(filename_list) < 25:
        rand_filename = filename_list
    else:
        rand_filename = random.sample(filename_list, 25)
    print(f"Displaying list of image(s): {rand_filename}")
    fig = plt.figure(figsize=(20, 20))
    for index, file in enumerate(rand_filename):
        tree = ET.parse(os.path.join(XML_PATH, f"{file}.xml"))
        root = tree.getroot()
        result = []  # store different bbox within the same file (image)
        for object in root.findall("object"):
            name = object.find("name").text
            for value in object.findall("bndbox"):
                xmin = int(value.find("xmin").text)
                ymin = int(value.find("ymin").text)
                xmax = int(value.find("xmax").text)
                ymax = int(value.find("ymax").text)
            result.append([name, xmin, ymin, xmax, ymax])
        # read the respective image
        img = cv2.imread(os.path.join(IMG_PATH, f"{file}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(result)):
            start = tuple(result[i][1:3])
            end = tuple(result[i][3:])
            colour = (255, 0, 255)  # magenta
            img = cv2.rectangle(img, start, end, colour, 2)
            img = cv2.putText(
                img,
                result[i][0],
                start,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colour,
                2,
            )
        fig.add_subplot(5, 5, index + 1)  # subplots of maximum 5 rows 5 columns
        plt.imshow(img)
        plt.title(f"{file}", fontsize=16)
    plt.tight_layout()
    plt.show()
