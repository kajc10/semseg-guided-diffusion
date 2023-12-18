import numpy as np
import cv2
from skimage.draw import polygon
import json
import os
from PIL import Image
from tqdm import tqdm
import argparse


def adjust_polygon_for_crop_and_scale(polygon, crop_size, final_size, original_width, original_height):
    # Adjust for center crop
    left = (original_width - crop_size) / 2
    top = (original_height - crop_size) / 2
    polygon[:, 0] -= left
    polygon[:, 1] -= top

    # Adjust for scale
    scale_factor = final_size / crop_size
    polygon *= scale_factor

    # Clip to new dimensions just to be safe
    polygon[:, 0] = np.clip(polygon[:, 0], 0, final_size - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, final_size - 1)

    return polygon


def create_mask_from_polygons_rescaled(height, width, labels, categories, crop_size, final_size):
    mask = np.zeros((final_size, final_size), dtype=np.uint8)
    for label in labels:
        category = label['category']
        if category in categories:
            cat_idx = categories.index(category)
            for poly2d in label['poly2d']:
                vertices = np.array(poly2d['vertices'])
                adjusted_vertices = adjust_polygon_for_crop_and_scale(vertices, crop_size, final_size, width, height)

                rr, cc = polygon(adjusted_vertices[:, 1], adjusted_vertices[:, 0])
                mask[rr, cc] = cat_idx
    return mask



def center_crop_and_resize(input_path, output_path, crop_size, final_size):
    try:
        img = Image.open(input_path)

        # Center crop
        left = (img.width - crop_size) / 2
        top = (img.height - crop_size) / 2
        right = (img.width + crop_size) / 2
        bottom = (img.height + crop_size) / 2
        img = img.crop((left, top, right, bottom))

        # Resize
        img = img.resize((final_size, final_size), Image.ANTIALIAS)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")


def process_images_and_semsegs(json_file_path, input_folder, output_folder, colormap, categories, crop_size, final_size):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    for item in tqdm(data, desc="Processing images and semsegs"):
        frame_name = item["name"]
        img_path = os.path.join(input_folder, f"{frame_name}") 

        # Process the original image
        output_img_path = os.path.join(output_folder,'images', f"{frame_name}")
        center_crop_and_resize(img_path, output_img_path, crop_size, final_size)

        # Process the segmentation mask
        mask = create_mask_from_polygons_rescaled(720, 1280, item['labels'], categories, crop_size, final_size)
        color_mask = colormap[mask].astype(np.uint8)
        output_mask_path = os.path.join(output_folder,'semsegs', f"{frame_name[:-4]}.png")

        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)  
        Image.fromarray(color_mask).save(output_mask_path)


def generate_colormap(num_classes):
    colormap = np.array([
        ((i * 123 + 7) % 256, (i * 456 + 8) % 256, (i * 789 + 9) % 256)
        for i in range(num_classes)
    ], dtype=np.int32)  # Ensure the dtype is int
    return colormap

def save_colormap_to_json(colormap, categories, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    color_dict = {categories[i]: list(map(int, colormap[i])) for i in range(len(categories))}
    with open(filepath, 'w') as json_file:
        json.dump(color_dict, json_file)


def load_colormap_from_json(filepath, categories):
    with open(filepath, 'r') as json_file:
        color_dict = json.load(json_file)
    return np.array([color_dict[cat] for cat in categories])

def generate_colorbook_image(categories, colormap, output_file):
    # Constants for the image layout
    RECT_WIDTH = 100
    RECT_HEIGHT = 50
    TEXT_OFFSET = (RECT_WIDTH + 10, int(RECT_HEIGHT / 2))
    IMAGE_WIDTH = RECT_WIDTH + 300
    IMAGE_HEIGHT = RECT_HEIGHT * len(categories)

    # Create an empty white image
    img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255

    # For each category, draw the rectangle and put the text
    for idx, category in enumerate(categories):
        top_left = (0, idx * RECT_HEIGHT)
        bottom_right = (RECT_WIDTH, (idx+1) * RECT_HEIGHT)

        color = tuple(reversed(colormap[idx].tolist()))
        cv2.rectangle(img, top_left, bottom_right, color, -1)
        cv2.putText(img, category, (TEXT_OFFSET[0], TEXT_OFFSET[1] + idx * RECT_HEIGHT), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(output_file, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and segmentation masks.')

    parser.add_argument('--subset', type=str, default='val', help='Subset of the dataset to process')
    parser.add_argument('--final_size', type=int, default=256, help='Final size of the images and masks')
    parser.add_argument('--input_folder', type=str, default='./data/bdd_fullres_10k_images/val', help='Input folder containing images')
    parser.add_argument('--output_folder', type=str, default='./data/bdd128x128/semsegval_images_masks', help='Output folder for processed images and masks')
    parser.add_argument('--colormap_path', type=str, default='./data/bdd128x128/colormap128x128val.json', help='Path to the colormap JSON file')
    parser.add_argument('--use_existing_colormap', type=bool, default=False, help='Flag to use existing colormap')
    parser.add_argument('--middle_crop_size', type=int, default=720, help='Size of the middle crop from the original image')

    args = parser.parse_args()

    # Categories initialization
    categories = [
        'background', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation',
        'terrain', 'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]

    if args.use_existing_colormap and os.path.exists(args.colormap_path):
        colormap = np.array(load_colormap_from_json(args.colormap_path, categories))
        generate_colorbook_image(categories, colormap, f'./data/bdd{args.final_size}x{args.final_size}/colorbook.png')
        print('loaded colormap from json')
    else:
        colormap = generate_colormap(len(categories))
        save_colormap_to_json(colormap.tolist(), categories, args.colormap_path)
        generate_colorbook_image(categories, colormap, f'./data/bdd{args.final_size}x{args.final_size}/colorbook.png')
        print('generated colormap')

    # Call the processing function
    json_file_path = f'./data/sem_seg/polygons/sem_seg_{args.subset}.json'
    process_images_and_semsegs(json_file_path, args.input_folder, args.output_folder, colormap, categories, args.middle_crop_size, args.final_size)