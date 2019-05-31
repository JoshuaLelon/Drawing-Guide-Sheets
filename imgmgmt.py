import sys
from PIL import Image, ImageOps
import os
import random 
import glob

##########  

FULL_IMG_PATH = 'full/'
IMAGES_PATH = 'images/'

  ### <pic_id>_<complexity>.png

def generate_trace_images(num_pics_per_page, difficulty_level):
    assert(num_pics_per_page in [1, 4, 9, 16])
    image_names = get_n_rand_pics_of_difficulty(num_pics_per_page, difficulty_level)
    full_tracer_image, full_dilated_image = get_full_images(image_names, num_pics_per_page)
    img_with_border.save(FULL_IMG_PATH + 'full_tracer_image.png')
    img_with_border.save(FULL_IMG_PATH + 'full_dilated_image.png')

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

def add_border_to_image(image_name, border_size=2):
    img = Image.open(image_name)
    img_with_border = ImageOps.expand(img, border=border_size, fill='black')
    img_with_border.save(image_name) # replaces original image
