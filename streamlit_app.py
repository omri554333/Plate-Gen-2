import streamlit as st
import numpy as np
import os
import shutil
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import gc
import random
import cv2
import math



def generate_strewbarrie():
    folder_path = os.path.join(os.getcwd(), 'new plan/colored strwbarries')
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


def generate_rose():
    path = 'new plan/colored flowers'
    folder_path = os.path.join(os.getcwd(), path)
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


def generate_dot():
    path = 'new plan/colored dots'
    folder_path = os.path.join(os.getcwd(), path)
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


def generate_small_flower():
    path = 'new plan/colored small flowers'
    folder_path = os.path.join(os.getcwd(), path)
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


def generate_blueberrie():
    path = 'new plan/colored bluebarries'
    folder_path = os.path.join(os.getcwd(), path)
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


def generate_redberrie():
    path = 'new plan/colored redbarries'
    folder_path = os.path.join(os.getcwd(), path)
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


def generate_leaf():
    path = 'new plan/colored leaf'
    folder_path = os.path.join(os.getcwd(), path)
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img


# Function to draw a vine with small branches
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    no_bg, _ = remove_background(rotated)
    return no_bg


def draw_vine(draw, image, x, y, length, angle, width, prctg, times=0, dir=1):
    max_times = 16
    # Draw rose at the end of the vine
    if times == max_times:
        offset = 20
        if dir == 1:
            x = x + 50
        else:
            x = x - 50
        srbr_img = generate_strewbarrie()
        srbr_img = scale_image(srbr_img, 1.2)
        if random.uniform(0, 1) < prctg:
            srbr_img = add_scribble_effect_on_object(srbr_img)
        srbr_img_pil = Image.fromarray(srbr_img)
        image.paste(srbr_img_pil, (int(x - srbr_img_pil.width // 2), int(y - offset - srbr_img_pil.height // 2)))
        return

    x_end = x + length * math.cos(math.radians(angle))
    y_end = y - length * math.sin(math.radians(angle))
    draw.line((x, y, x_end, y_end), fill=(20, 127, 168), width=width)

    if times == 9 and dir == -1:
        # Calculate rose placement coordinates based on the direction
        offset = 70  # Distance from the vine
        if dir == -1:
            rose_x = x_end + offset
        else:
            # rose_x = x_end + offset
            pass

        rose_y = y_end

        # Draw rose at intervals along the vine
        srbr_img = generate_strewbarrie()
        srbr_img = scale_image(srbr_img, 1.2)
        if random.uniform(0, 1) < prctg:
            srbr_img = add_scribble_effect_on_object(srbr_img)
        srbr_img_pil = Image.fromarray(srbr_img)
        image.paste(srbr_img_pil, (int(rose_x - srbr_img_pil.width // 2), int(rose_y - srbr_img_pil.height // 2)))

    new_length = length * 0.9
    new_angle = angle + random.uniform(0, 8) * dir
    new_width = int(width * 0.95)  # Maintain a good tapering effect

    if times == 5:
        new_length_1 = 50  # Length for the new vine segment
        new_angle_1 = 90 + random.uniform(0, 10) * -1  # Random angle for the new vine segment
        new_width_1 = 10  # Width for the new vine segment
        dir = 1
        draw_vine(draw, image, x_end, y_end, new_length_1, new_angle_1, new_width_1, prctg, times + 1, -1)

    draw_vine(draw, image, x_end, y_end, new_length, new_angle, new_width, prctg, times + 1, dir)


# Function to generate a vine image starting from the bottom center
def generate_vine_image(diameter, prctg):
    segments_drawn = 0  # Reset global counter
    image = Image.new("RGB", (diameter, diameter), "white")
    draw = ImageDraw.Draw(image)

    start_x = diameter // 2
    start_y = diameter - 20  # Start slightly above the bottom edge
    initial_length = 100
    initial_angle = 90  # Angle in degrees, pointing upwards
    initial_width = 15  # Increased initial width

    draw_vine(draw, image, start_x, start_y, initial_length, initial_angle, initial_width, prctg)

    return image


def remove_background(img):
    img_array = np.array(img)

    if img_array.shape[2] == 4:  # Check if the image has an alpha channel
        b, g, r, a = cv2.split(img_array)
        img_rgb = cv2.merge((b, g, r))
        mask = a
    else:
        img_rgb = img_array
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours and filter based on area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust area threshold as needed
            cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)

    mask_3c = cv2.cvtColor(mask_filled, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(img_rgb, mask_3c)

    # Create a white background
    white_bg = np.zeros_like(img_rgb)

    # Combine the result with the white background
    inverted_mask_3c = cv2.bitwise_not(mask_3c)
    white_part = cv2.bitwise_and(white_bg, inverted_mask_3c)
    final_result = cv2.add(result, white_part)

    return final_result, mask_filled


def rotate_image(image, angle=0):
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Get the rotation matrix for the desired angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the bounding box of the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to consider the new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Rotate the image with the new bounding dimensions
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
    return rotated_image


def create_white_circle(diameter):
    circle_img = np.ones((diameter, diameter, 3), dtype=np.uint8) * 255
    cv2.circle(circle_img, (diameter // 2, diameter // 2), diameter // 2, (0, 0, 0), -1)
    return circle_img


# def generate_set_of_objects(generator_function, num_objects):
#     objects = []
#     for _ in range(num_objects):
#         obj = generator_function()
#         # obj,_=remove_background(obj)
#         angle = random.randint(0, 360)  # Random angle between 0 and 360 degrees
#         # rotated_obj = rotate_image(obj, angle)
#         rotated_obj,_=remove_background(obj,angle)
#         objects.append(rotated_obj)
#     return objects

def generate_set_of_objects(generator_function, num_objects):
    objects = []
    for _ in range(num_objects):
        obj = generator_function()
        angle = random.randint(0, 360)  # Random angle between 0 and 360 degrees
        rotated_obj = rotate_image(obj, angle)

        # Reapply background removal after rotation to get the correct mask
        rotated_obj, _ = remove_background(rotated_obj)
        objects.append(rotated_obj)
    return objects


def scale_image(image, scale):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled


def create_custom_mask(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use edge detection (Canny) to find the object edges
    edges = cv2.Canny(blurred, 50, 150)

    # Use morphological operations to close gaps in edges and fill the object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours based on the detected edges
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    mask = np.zeros_like(gray)

    # Draw the contours onto the mask
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small noise contours by area threshold
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask


def add_scribble_effect_on_object(img, scribble_intensity=70, scribble_color=(255, 255, 255)):
    # Create a custom mask for the object
    object_mask = create_custom_mask(img)

    # Create a blank image for the scribble effect
    scribble_img = np.zeros_like(img)  # Changed to zeros to match object area only

    # Scribble effect parameters
    height, width = img.shape[:2]
    num_scribbles = int((height * width) * (scribble_intensity / 10000))  # Adjust the scribble density

    obj = generate_set_of_objects(generate_rose, 1)[0]

    # Random scribbles within the object mask
    for _ in range(num_scribbles):
        # Random starting point
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)

        # Only draw if the point is within the object mask
        if object_mask[y1, x1] != 0:
            # Random ending point close to the starting point to create a short scribble
            x2, y2 = np.clip(x1 + np.random.randint(-15, 15), 0, width - 1), np.clip(y1 + np.random.randint(-15, 15), 0,
                                                                                     height - 1)

            # Random thickness
            thickness = np.random.randint(1, 4)

            # Draw the line with the specified scribble color
            cv2.line(scribble_img, (x1, y1), (x2, y2), scribble_color, thickness)

    # Apply the scribble effect only to the object
    scribble_with_object = cv2.bitwise_and(scribble_img, scribble_img, mask=object_mask)

    # Combine the scribble effect with the original image, emphasizing the scribbles
    combined_image = cv2.addWeighted(img, 0.9, scribble_with_object, 0.5, 0)  # Adjust the blending ratio if needed

    return combined_image


def place_object(base_img, obj, x, y, mask, vine_mask, prctg):
    if random.uniform(0, 1) < prctg:
        obj = add_scribble_effect_on_object(obj)

    obj_h, obj_w, _ = obj.shape
    x_start = x - obj_w // 2
    y_start = y - obj_h // 2

    if 0 <= x_start < base_img.shape[1] - obj_w and 0 <= y_start < base_img.shape[
        0] - obj_h:  # <-----------------------------------
        # Check if the area is empty and not on vine
        if np.all(base_img[y_start:y_start + obj_h, x_start:x_start + obj_w] == 0) and \
                np.all(mask[y_start:y_start + obj_h, x_start:x_start + obj_w] == 0) and \
                np.all(vine_mask[y_start:y_start + obj_h, x_start:x_start + obj_w] == 0):
            if random.uniform(0, 1) < prctg:
                obj = add_scribble_effect_on_object(obj)
            base_img[y_start:y_start + obj_h, x_start:x_start + obj_w] = obj
            mask[y_start:y_start + obj_h, x_start:x_start + obj_w] = 1
            return True
    return False


def place_objects_in_segment(base_img, mask, vine_mask, objects, center, inner_radius, outer_radius, scales, prctg,
                             att=1000):
    attempts = att
    for _ in range(attempts):
        obj = random.choice(objects)  # Pick a random object each time
        scl = random.choice(scales)  # Pick a random scale each time
        scaled_obj = scale_image(obj, scl)
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(inner_radius, outer_radius)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        place_object(base_img, scaled_obj, x, y, mask, vine_mask, prctg)


def place_objects_in_segments(base_img, diameter, objects, vine_mask, prctg):
    center = (diameter // 2, diameter // 2)
    mask = np.zeros((diameter, diameter), dtype=np.uint8)

    radius1 = diameter // 5
    radius2 = 2 * diameter // 6
    radius3 = diameter // 2

    # Divide objects into three sets for each segment
    num_objects_per_segment = len(objects) // 3
    objects_segment1 = objects[2]
    objects_segment2 = objects[4]
    objects_segment3 = objects[1] + objects[3]
    objects_segment4 = objects[5]
    objects_segment5 = objects[0]

    # Place objects in each segment
    place_objects_in_segment(base_img, mask, vine_mask, objects_segment1, center, 0, radius1, [1.4], prctg)
    place_objects_in_segment(base_img, mask, vine_mask, objects_segment2, center, radius1, radius2, [0.7], prctg)
    place_objects_in_segment(base_img, mask, vine_mask, objects_segment3, center, radius2, radius3, [0.5], prctg, 100)
    place_objects_in_segment(base_img, mask, vine_mask, objects_segment5, center, radius2, radius3, [0.3], prctg, 100)
    place_objects_in_segment(base_img, mask, vine_mask, objects_segment4, center, 0, radius3, [0.1], prctg, 20)

    return base_img


def create_border(diameter, border_width, colors):
    border = np.zeros((diameter, diameter, 3), dtype=np.uint8)
    radius = diameter // 2
    for i, color in enumerate(colors):
        cv2.circle(border, (radius, radius), radius - i * border_width, color, border_width)
    return border


def Generate_Plate(prctg, ifborder=False):
    # Step 1: Create the black circle base
    diameter = 1024
    border_width = 20
    border_radius = (diameter // 2) - border_width  # Define the border radius

    white_circle = create_white_circle(diameter)

    # Step 2: Generate the vine starting from the bottom center
    vine_image = generate_vine_image(diameter, prctg)

    # Convert PIL image to OpenCV format and remove background
    vine_image_cv = np.array(vine_image)
    vine_image_cv, vine_mask = remove_background(vine_image_cv)

    # Overlay the vine on the white circle
    mask = (white_circle != [255, 255, 255]).all(axis=2)

    # Overlay the vine image on the white circle using the mask
    white_circle[mask] = vine_image_cv[mask]

    # Step 3: Generate 10 variations of each object
    roses = generate_set_of_objects(generate_rose, 5)
    grapes = generate_set_of_objects(generate_small_flower, 5)
    leafs = generate_set_of_objects(generate_leaf, 5)
    blueberries = generate_set_of_objects(generate_blueberrie, 5)
    dots = generate_set_of_objects(generate_dot, 5)
    redbarries = generate_set_of_objects(generate_redberrie, 5)

    # Combine objects into a list for the central motif
    objects = [redbarries, grapes, leafs, blueberries, roses, dots]
    #         [0           ,1     ,2        ,3        ,4      , 5]

    # Place objects in segments on the image with vines
    white_circle_with_objects = place_objects_in_segments(white_circle, diameter, objects, vine_mask, prctg)

    # Step 4: Apply the border
    colors = [(0, 0, 255), (0, 255, 255), (0, 0, 0)]  # Updated border colors
    border_diameter = diameter + 2 * border_width
    border = create_border(border_diameter, border_width, colors)

    # Combine the black circle with border
    final_design = np.ones((border_diameter, border_diameter, 3), dtype=np.uint8) * 255
    center = (border_diameter // 2, border_diameter // 2)
    x_start = center[0] - diameter // 2
    y_start = center[1] - diameter // 2
    final_design[y_start:y_start + diameter, x_start:x_start + diameter] = white_circle_with_objects

    # Apply the border
    # Create a mask where the border is not black
    if ifborder:
        mask = (border != [0, 0, 0]).any(axis=2)

        # Overlay the border on the final design using the mask
        final_design[mask] = border[mask]
    final_design = cv2.cvtColor(final_design, cv2.COLOR_BGR2RGB)

    return final_design


# Title of the app
st.title("Plate Generator")

# Load the images
image1 = Image.open("1-01.jpg")
image2 = Image.open("1-02.jpg")
image3 = Image.open("1-03.jpg")

# Display the image at the top
st.image(image3, use_column_width=True)

# Create three columns
col1, col2, col3 = st.columns([1, 2, 1])

# Add some vertical space above the slider
with col1:
    st.image(image1, use_column_width=True)

with col2:
    st.write("")  # Adds an empty line for spacing
    st.write("")  # Adds more space; adjust as needed
    st.write("")
    st.write("")

    percentage = st.slider("Select Percentage", 0, 100, 50)

with col3:
    st.image(image2, use_column_width=True)

# Button to generate the plate
if st.button("Generate Plate"):
    st.write(f"Generating plate with {percentage}%...")
    generated_image = Generate_Plate(percentage / 100)

    # Convert to PIL format for display
    generated_image_pil = Image.fromarray(generated_image)
    st.image(generated_image_pil, use_column_width=True)
