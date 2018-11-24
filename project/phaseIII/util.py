#! /bin/usr/python3.6

from time import time
from functools import wraps
from os import path
from PIL import Image

##
# Add as decorator to any function to time how long the function takes to execute.
#
# Decorates must come after static declarations. For example:
#   @staticmethod
#   @timed
#   def foo():
#
def timed(f):
    @wraps(f)
    def wrapper(*argv, **kwargs):
        start = time()
        return_val = f(*argv, **kwargs)
        print(str(f.__name__) + " executed in " + str(time() - start))
        return return_val
    return wrapper



def show_images(images, db):
    """
    Displays a list of image id's by opening them up in individual windows.
    :param list images: list of image id's.
    :param Database db:
    """
    files = get_img_files(images, db)
    show_files(files)



def get_img_file(image, db):
    """
    Gets the file location for the image it is given.
    :param int image: id of image to retrieve.
    :param Database db: Database object.
    """
    img_dir = db.source
    if img_dir == None:
        raise ValueError('Cannot locate file without a base path. This method looks for it at \
        db.source, which is not set. This should be set by the loader during DB construction!')
    img_dir = path.join(img_dir, 'img')
    # get location title.
    loc_id = db.get_img_loc(int(image))
    if loc_id == None:
        raise ValueError('The image %s could not be found' % image)
    loc = db.get_location(loc_id)
    title = loc['title']
    # add to file name
    img_dir = path.join(img_dir, title, str(image) + '.jpg')
    return img_dir


def get_img_files(images, db):
    """
    Gets the file location for the images it is given.
    :param list images: list of int image ids.
    :param Database db: Database obj.
    """
    img_dir = db.source
    if img_dir == None:
        raise ValueError('Cannot locate file without a base path. This method looks for it at \
        db.source, which is not set. This should be set by the loader during DB construction!')
    img_dir = path.join(img_dir, 'img') 
    locs = db.get_img_locs(images)
    titles = db.get_location_titles()
    returnval = []
    for image in images:
        loc = locs[image]
        if loc is None:
            raise ValueError('The image %s could not be found' % image)
        returnval.append(path.join(img_dir, titles[loc], str(image) + '.jpg'))
    return returnval


def show_file(file_location):
    """
    Displays an image to screen by opening it up in a separate window.
    :param str file_location: path to the image.
    """
    img = Image.open(file_location)
    img.show()


def show_files(file_locations):
    """
    Displays a set of images to screen by opening them up in individual windows.
    :param list file_locations: list of paths to images.
    """
    for file_loc in file_locations:
        show_image(file_loc)


def show_image(file_location):
    """
    Displays an image to screen by opening it up in a separate window.
    :param str file_location: path to the image.
    """
    img = Image.open(file_location)
    img.show()