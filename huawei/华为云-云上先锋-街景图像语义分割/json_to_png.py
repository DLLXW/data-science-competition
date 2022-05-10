
# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt

# Image processing
from PIL import Image
from PIL import ImageDraw

# cityscapes imports
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import name2label
#
# a label and all meta information
from collections import namedtuple
from tqdm import tqdm
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])
labels = [
    #       name  id        trainId     category     catId     hasInstances   ignoreInEval   color
    Label('void', 255,            255,    'void',         0,          False,      True,       (0, 0, 0)),
    Label('flat', 0,            0,      'flat',         1,          False,      False,      (128, 64, 128)),
    Label('human', 1,           1,      'human',        2,          True,       False,      (220, 20, 60)),
    Label('vehicle', 2,         2,      'vehicle',      3,          True,       False,      (0, 0, 142)),
    Label('construction', 3,    3,      'construction', 4,          False,      False,      (70, 70, 70)),
    Label('object', 4,          4,      'object',       5,          False,       False,      (153, 153, 153)),
    Label('nature', 5,          5,      'nature',       6,          False,      False,      (107, 142, 35)),
    Label('sky', 6,            6,      'sky',          7,          False,      False,      (70, 130, 180)),


]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}


#
# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print('Reads labels as polygons in JSON format and converts them to label images,')
    print('where each pixel has an ID that represents the ground truth label.')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')


# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)


# Convert the given annotation to a label image
def createLabelImage(annotation, encoding, outline=None):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background
    if encoding == "ids":
        background = name2label['void'].id
    elif encoding == "trainIds":
        background = name2label['void'].trainId
    elif encoding == "color":
        background = name2label['void'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(labelImg)

    # loop over all objects
    for obj in annotation.objects:
        label = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # If the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        if (not label in name2label) and label.endswith('group'):
            label = label[:-len('group')]
        #

        if not label in name2label:
            printError("Label '{}' not known.".format(label))

        # If the ID is negative that polygon should not be drawn
        if name2label[label].id < 0:
            continue

        if encoding == "ids":
            val = name2label[label].id
        elif encoding == "trainIds":
            val = name2label[label].trainId
        elif encoding == "color":
            val = name2label[label].color

        try:
            if outline:
                drawer.polygon(polygon, fill=val, outline=outline)
            else:
                drawer.polygon(polygon, fill=val)
        except:
            print("Failed to draw polygon with label {}".format(label))
            raise

    return labelImg


# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the label image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
#     - "color"    : classes are encoded using the corresponding colors
def json2labelImg(inJson, outImg, encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    labelImg = createLabelImage(annotation, encoding)
    labelImg.save(outImg)


# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2labelImg'
def main(inJson,outImg):
    trainIds = False

    # inJson = '/media/ssd/2034.json'
    # outImg = '/media/ssd/2034_1.png'

    if trainIds:
        json2labelImg(inJson, outImg, "trainIds")
    else:
        json2labelImg(inJson, outImg,"trainIds")


# call the main method
if __name__ == "__main__":
    json_dir = '/media/ssd/huawei_seg_own/jsons/'
    mask_dir='/media/ssd/huawei_seg_own/huawei_data/mask'
    names = os.listdir(json_dir)
    for i in tqdm(range(1204,len(names))): #943,1119,1665有标注错误（0787.json,0568.json,0086.json）
        if i in [943,1204,1785]:
            print(names[i])
            continue
        name=names[i]
        per_path = os.path.join(json_dir, name)
        main(per_path,os.path.join(mask_dir,name[:-5]+'.png'))