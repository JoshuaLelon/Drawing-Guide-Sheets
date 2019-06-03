import sys
from PIL import Image, ImageOps
import os
import random 
import glob
import cv2
import math
import numpy as np

##########  

FULL_IMG_PATH = os.path.dirname(os.path.abspath(__file__)) + '/full/'
IMAGES_PATH = os.path.dirname(os.path.abspath(__file__)) + '/images/'

FULL_TRACEABLE_IMAGE_NAME = 'full_traceable_image.png'
FULL_TARGET_IMAGE_NAME = 'full_target_image.png'
FULL_BOILERPLATE_IMAGE_NAME = 'full_boilerplate_image.png'

LENGTH_OF_PAGE_IN_PIXELS = 3300
WIDTH_OF_PAGE_IN_PIXELS = 2550
INCREMENT = 200
BLACK = [0, 0, 0]

  ### <pic_id>_<complexity>.png

def generate_full_boilerplate_and_target_images(num_pics_per_page, difficulty_level, space_on_page, percentage_to_cut):
    assert(num_pics_per_page in [1, 4, 9, 16])
    image_names = get_n_rand_pics_of_difficulty(num_pics_per_page, difficulty_level)
    full_boilerplate_image, full_target_image = get_full_boilerplate_and_target_images(image_names, num_pics_per_page, space_on_page, percentage_to_cut)
    print(FULL_IMG_PATH + FULL_BOILERPLATE_IMAGE_NAME)
    cv2.imwrite(FULL_IMG_PATH + FULL_BOILERPLATE_IMAGE_NAME, full_boilerplate_image)
    cv2.imwrite(FULL_IMG_PATH + FULL_TARGET_IMAGE_NAME, full_target_image)

def get_full_boilerplate_and_target_images(image_names, num_pics_per_page, space_on_page, percentage_to_cut):
    images = read_images(image_names)
    full_target_image = generate_full_regular_image(images, space_on_page)
    full_boilerplate_image = create_boilerplate_full_image(images, space_on_page, percentage_to_cut)
    return full_boilerplate_image, full_target_image

def read_images(image_names):
    images = []
    for image_name in image_names:
        img = cv2.imread(image_name)
        images.append(img)
    return images

def generate_full_regular_image(images, space_on_page):
    bordered_images = []
    for image in images:
        with_border = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
        bordered_images.append(with_border)
    return create_full_image(bordered_images, space_on_page)

def create_boilerplate_full_image(images, space_on_page, percentage_to_cut):
    bordered_cut_images = []
    for image in images:
        cut_img = cut_n_percent_of_image(image, percentage_to_cut)
        with_border = cv2.copyMakeBorder(cut_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
        bordered_cut_images.append(with_border)
    return create_full_image(bordered_cut_images, space_on_page)

def cut_n_percent_of_image(image, percentage_to_cut):
    where_to_cut = random.randint(0, 3)
    y_offset = None
    x_offset = None
    image_cover = None
    if where_to_cut == 0: # cut the left chunk
        new_x = int((image.shape[1] * percentage_to_cut) / 100)
        image_cover = create_pure_white_img(image.shape[0], new_x)
        y_offset = 0
        x_offset = 0
    elif where_to_cut == 1: # cut the top chunk
        new_y = int((image.shape[0] * percentage_to_cut) / 100)
        image_cover = create_pure_white_img(new_y, image.shape[1])
        y_offset = 0
        x_offset = 0
    elif where_to_cut == 2: # cut the right chunk
        new_x = int((image.shape[1] * percentage_to_cut) / 100)
        image_cover = create_pure_white_img(image.shape[0], new_x)
        y_offset = 0
        x_offset = image.shape[1] - new_x
    else:                   # cut the bottom chunk
        new_y = int((image.shape[0] * percentage_to_cut) / 100)
        image_cover = create_pure_white_img(new_y, image.shape[1])
        y_offset = image.shape[0] - new_y
        x_offset = 0
    # print("image shape: ", image.shape)
    # print("image_cover shape: ", image_cover.shape)
    # print("y_offset: ", y_offset)
    # print("x_offset: ", x_offset)
    image[y_offset:y_offset + image_cover.shape[0], x_offset:x_offset + image_cover.shape[1]] = image_cover
    return image

def create_pure_white_img(y, x):
    return np.ones((y, x, 3), np.uint8) * 255

###############################################################

def generate_just_traceable(num_pics_per_page, difficulty_level, space_on_page): # space_on_page = 'minimal', 'moderate', 'maximal'
    assert(num_pics_per_page in [1, 4, 9, 16])
    image_names = get_n_rand_pics_of_difficulty(num_pics_per_page, difficulty_level)
    full_traceable_image = get_full_traceable_image(image_names, num_pics_per_page, space_on_page)
    cv2.imwrite(FULL_IMG_PATH + FULL_TRACEABLE_IMAGE_NAME, full_traceable_image)

def get_n_rand_pics_of_difficulty(n, difficulty_level):
    image_names_list = get_all_image_names()
    image_names_of_proper_difficulty = []
    for image_name in image_names_list:
        filename, file_extension = os.path.splitext(image_name)
        complexity = filename.split("_", 1)[1]
        if int(complexity) == difficulty_level:
            image_names_of_proper_difficulty.append(image_name)
    random.shuffle(image_names_of_proper_difficulty)
    print("Picked images: ")
    print(image_names_of_proper_difficulty[0:n])
    return image_names_of_proper_difficulty[0:n]

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
    kernel = np.zeros((2,2), np.uint8) 
    img_erosion = cv2.erode(cv2_image, kernel, iterations=1)
    with_border = cv2.copyMakeBorder(img_erosion, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)
    return with_border

def create_full_image(images, space_on_page):
    image_frame_size = WIDTH_OF_PAGE_IN_PIXELS
    if space_on_page == 'minimal':
        image_frame_size = WIDTH_OF_PAGE_IN_PIXELS - 2 * INCREMENT
    if space_on_page == 'moderate':
        image_frame_size = WIDTH_OF_PAGE_IN_PIXELS - INCREMENT
    
    images_per_side = int(math.sqrt(len(images)))
    single_image_len = int(image_frame_size / images_per_side)
    target_image_grid = np.zeros((image_frame_size, image_frame_size, 3), np.uint8)
    for x in range(images_per_side):
        for y in range(images_per_side):
            img = images.pop()
            img = cv2.resize(img, (single_image_len, single_image_len)) 
            x_offset = x * single_image_len
            y_offset = y * single_image_len
            target_image_grid[y_offset:y_offset + single_image_len, x_offset:x_offset + single_image_len] = img

    full_image = create_pure_white_img(LENGTH_OF_PAGE_IN_PIXELS, WIDTH_OF_PAGE_IN_PIXELS)
    x_offset = int((WIDTH_OF_PAGE_IN_PIXELS - image_frame_size) / 2)
    y_offset = int((LENGTH_OF_PAGE_IN_PIXELS - image_frame_size) / 2)
    full_image[y_offset:y_offset + target_image_grid.shape[0], x_offset:x_offset + target_image_grid.shape[1]] = target_image_grid
    return full_image

if __name__ == "__main__":
    if len(sys.argv) == 5:
        print("Number of pictures per page: ", sys.argv[1])
        print("Difficulty level: ", sys.argv[2])
        print("Size of images on page: ", sys.argv[3]) # minimal, moderate, maximal
        print("Percentage to cut: ", sys.argv[4])
        print("----")
        print("Generating boilerplate and target images...")
        generate_full_boilerplate_and_target_images(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) == 4:
        print("Number of pictures per page: ", sys.argv[1])
        print("Difficulty level: ", sys.argv[2])
        print("Size of images on page: ", sys.argv[3]) # minimal, moderate, maximal
        print("----")
        print("Generating just traceable images...")
        generate_just_traceable(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    else:
        print("----")
        print("Generating boilerplate and target images with pictures: 9, difficulty: 3, moderate spacing, and 60 percent cut.")
        generate_full_boilerplate_and_target_images(9, 3, 'moderate', 60)
    