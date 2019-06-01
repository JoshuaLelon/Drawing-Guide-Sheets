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

LENGTH_OF_PAGE_IN_PIXELS = 3300
WIDTH_OF_PAGE_IN_PIXELS = 2550
INCREMENT = 200

  ### <pic_id>_<complexity>.png

def generate_trace_images(num_pics_per_page, difficulty_level):
    assert(num_pics_per_page in [1, 4, 9, 16])
    image_names = get_n_rand_pics_of_difficulty(num_pics_per_page, difficulty_level)
    full_traceable_image, full_target_image = get_both_full_images(image_names, num_pics_per_page)
    full_traceable_image.save(FULL_IMG_PATH + FULL_TRACEABLE_IMAGE_NAME)
    full_target_image.save(FULL_IMG_PATH + FULL_TARGET_IMAGE_NAME)

def get_full_images(image_names, num_pics_per_page):

    images = map(Image.open, image_names)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

    new_im.save('test.jpg')

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
            x_offset = x * img.shape[1]
            y_offset = y * img.shape[0]
            target_image_grid[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img


    

    full_image = np.ones((LENGTH_OF_PAGE_IN_PIXELS, WIDTH_OF_PAGE_IN_PIXELS, 3), np.uint8)
    x_offset = int((WIDTH_OF_PAGE_IN_PIXELS - image_frame_size) / 2)
    y_offset = int((LENGTH_OF_PAGE_IN_PIXELS - image_frame_size) / 2)
    full_image[y_offset:y_offset + target_image_grid.shape[0], x_offset:x_offset + target_image_grid.shape[1]] = target_image_grid

    return full_image

def add_border_to_image(image_name, border_size=2):
    img = Image.open(image_name)
    img_with_border = ImageOps.expand(img, border=border_size, fill='black')
    img_with_border.save(image_name) # replaces original image
