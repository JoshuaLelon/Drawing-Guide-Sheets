import sys
from PIL import Image, ImageOps
import os
import random 
import glob
import cv2
import math
import numpy as np

##########  

FULL_IMG_PATH = 'full/'
IMAGES_PATH = 'images/'

FULL_TRACEABLE_IMAGE_NAME = 'full_traceable_image.png'
FULL_TARGET_IMAGE_NAME = 'full_target_image.png'
FULL_BOILERPLATE_IMAGE_NAME = 'full_boilerplate_image.png'

LENGTH_OF_PAGE_IN_PIXELS = 3300
WIDTH_OF_PAGE_IN_PIXELS = 2550
INCREMENT = 200

  ### <pic_id>_<complexity>.png

def generate_full_boilerplate_and_target_images(num_pics_per_page, difficulty_level, space_on_page, percentage_to_cut):
    assert(num_pics_per_page in [1, 4, 9, 16])
    image_names = get_n_rand_pics_of_difficulty(num_pics_per_page, difficulty_level)
    full_boilerplate_image, full_target_image = get_full_boilerplate_and_target_images(image_names, num_pics_per_page, space_on_page, percentage_to_cut)
    full_boilerplate_image.save(FULL_IMG_PATH + FULL_BOILERPLATE_IMAGE_NAME)
    full_target_image.save(FULL_IMG_PATH + FULL_TARGET_IMAGE_NAME)

def get_full_boilerplate_and_target_images(image_names, num_pics_per_page, space_on_page, percentage_to_cut):
    images = read_images(image_names)
    full_target_image = create_full_image(images, space_on_page)
    full_boilerplate_image = create_boilerplate_full_image(images, space_on_page, percentage_to_cut)
    return full_boilerplate_image, full_target_image

def read_images(image_names):
    images = []
    for image_name in image_names:
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images

def create_boilerplate_full_image(images, space_on_page, percentage_to_cut):
    return full_boilerplate_image
###############################################################

def generate_just_traceable(num_pics_per_page, difficulty_level, space_on_page): # space_on_page = 'minimal', 'moderate', 'maximal'
    assert(num_pics_per_page in [1, 4, 9, 16])
    image_names = get_n_rand_pics_of_difficulty(num_pics_per_page, difficulty_level)
    full_traceable_image = get_full_traceable_image(image_names, num_pics_per_page, space_on_page)
    full_traceable_image.save(FULL_IMG_PATH + FULL_TRACEABLE_IMAGE_NAME)

def get_n_rand_pics_of_difficulty(n, difficulty_level):
    image_names_list = get_all_image_names()
    image_names_of_proper_difficulty = []
    for image_name in image_names_list:
        filename, file_extension = os.path.splitext(image_name)
        complexity = filename.split("_", 1)[1]
        if int(complexity) == difficulty_level:
            image_names_of_proper_difficulty.append(image_name)
    random.shuffle(image_names_of_proper_difficulty)
    return image_names_of_proper_difficulty[0:num_pics_per_page]

def get_all_image_names():
    os.chdir(IMAGES_PATH)
    image_names = []
    for file in glob.glob("*.png"):
        image_names.append(file)
    return image_names

def get_full_traceable_image(image_names, num_pics_per_page, space_on_page):
    bordered_eroded_images = []
    for image_name in image_names:
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        prepped_img = erode_and_border(img)
        bordered_eroded_images.append(prepped_img)
    return create_full_image(bordered_eroded_images, space_on_page)
    
def erode_and_border(cv2_image):
    kernel = np.ones((2,2), np.uint8) 
    img_erosion = cv2.erode(cv2_image, kernel, iterations=1)
    BLACK = [0, 0, 0]
    with_border = cv2.copyMakeBorder(img_erosion, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
    return with_border

def create_full_image(images, space_on_page):
    image_frame_size = WIDTH_OF_PAGE_IN_PIXELS
    if space_on_page == 'minimal':
        image_frame_size = WIDTH_OF_PAGE_IN_PIXELS - 2 * INCREMENT
    if space_on_page == 'moderate':
        image_frame_size = WIDTH_OF_PAGE_IN_PIXELS - INCREMENT
    
    images_per_side = math.sqrt(len(images))
    single_image_len = int(image_frame_size / images_per_side)
    target_image_grid = np.zeros((image_frame_size, image_frame_size, 3), np.uint8)
    for x in range(images_per_side):
        for y in range(images_per_side):
            img = images.pop()
            img = cv2.resize(img, (single_image_len, single_image_len)) 
            x_offset = x * single_image_len
            y_offset = y * single_image_len
            target_image_grid[y_offset:y_offset + single_image_len, x_offset:x_offset + single_image_len] = img

    full_image = np.ones((LENGTH_OF_PAGE_IN_PIXELS, WIDTH_OF_PAGE_IN_PIXELS, 3), np.uint8)
    x_offset = int((WIDTH_OF_PAGE_IN_PIXELS - image_frame_size) / 2)
    y_offset = int((LENGTH_OF_PAGE_IN_PIXELS - image_frame_size) / 2)
    full_image[y_offset:y_offset + target_image_grid.shape[0], x_offset:x_offset + target_image_grid.shape[1]] = target_image_grid
    return full_image
