import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import gc
import random
import cv2
import math


def generate_strewbarrie():
    folder_path = 'new plan/colored strwbarries'
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img

def generate_rose():
    folder_path = 'new plan/colored flowers'
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img

def generate_dot():
    folder_path = 'new plan/colored dots'
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img

def generate_small_flower():
    folder_path = 'new plan/colored small flowers'
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img

def generate_blueberrie():
    folder_path = 'new plan/colored bluebarries'
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img

def generate_redberrie():
    folder_path = 'new plan/colored redbarries'
    file_list = os.listdir(folder_path)
    random_file = random.choice(file_list)
    img_path = os.path.join(folder_path, random_file)
    img = cv2.imread(img_path)
    return img
    
def generate_leaf():
    folder_path = 'new plan/colored leaf'
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
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    no_bg,_=remove_background(rotated)
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

    # # Display the final plate design
    # rescaled=scale_image(final_design, 0.21484375)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(final_design)
    # plt.axis('off')
    # plt.show()
    return final_design

# Call the function to generate and display the plate design
# plate=Generate_Plate(0)

# import tkinter as tk
# from tkinter import ttk

# def on_scale_change(event):
#     # Get the current value of the scale
#     value = percentage_var.get()
#     # Round to the nearest multiple of 10
#     rounded_value = round(value / 10) * 10
#     # Set the rounded value back to the scale
#     scale.set(rounded_value)
#     # Update the label with the rounded value
#     percentage_var.set(rounded_value)

# def on_button_click():
#     selected_percentage = percentage_var.get()
#     print(f"Selected Percentage: {selected_percentage}%")
#     Generate_Plate(selected_percentage/100)  

# # Create the main window
# root = tk.Tk()
# root.title("Plate Generator")
# root.geometry("600x400")  # Set the size of the window to 600x400 pixels

# # Create a frame to hold the meter and button
# frame = ttk.Frame(root, padding="20")
# frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
# frame.columnconfigure(0, weight=1)
# frame.rowconfigure(0, weight=1)

# # Create a variable to store the selected percentage
# percentage_var = tk.IntVar(value=0)

# # Create the meter (scale) in the middle of the screen
# scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal", length=400, variable=percentage_var)
# scale.grid(row=0, column=0, padx=20, pady=20)
# scale.bind("<ButtonRelease-1>", on_scale_change)  # Bind the scale change event
# scale.bind("<Motion>", on_scale_change)  # Bind the scale change event

# # Create a label to show the selected percentage
# percentage_label = ttk.Label(frame, textvariable=percentage_var, font=("Helvetica", 14))
# percentage_label.grid(row=1, column=0, pady=20)

# # Create the button below the meter
# button = ttk.Button(frame, text="Submit", command=on_button_click)
# button.grid(row=2, column=0, pady=20)

# # Ensure the frame expands to fill the window
# root.grid_columnconfigure(0, weight=1)
# root.grid_rowconfigure(0, weight=1)

# # Run the application
# root.mainloop()

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


def on_scale_change(event):
    value = percentage_var.get()
    rounded_value = round(value / 10) * 10
    scale.set(rounded_value)


def on_scale_click(event):
    # Calculate the value based on the click position
    new_value = int((event.x / scale.winfo_width()) * 100)  # Adjust range if necessary
    scale.set(new_value)  # Set the scale to the calculated value
    percentage_var.set(new_value)  # Update the variable


def on_button_click():
    # Disable the button to prevent further clicks while processing
    button.config(state=tk.DISABLED)

    selected_percentage = percentage_var.get()
    # print(f"Selected Percentage: {selected_percentage}%")

    # Generate the plate image
    generated_image = Generate_Plate(selected_percentage / 100)

    # Resize the generated image for display
    generated_image = cv2.resize(generated_image, (400, 400), interpolation=cv2.INTER_LANCZOS4)
    generated_image = Image.fromarray(generated_image)
    generated_photo = ImageTk.PhotoImage(generated_image)

    # Update the label with the new image
    generated_image_label.config(image=generated_photo)
    generated_image_label.image = generated_photo  # Keep a reference to avoid garbage collection

    # Re-enable the button after processing is complete
    button.config(state=tk.NORMAL)


# Create the main window
root = tk.Tk()
root.title("Plate Generator בשיתוף פעולה עם המחלקה למתמתיקה שימושית")
root.geometry("1980x1020")  # Set the size of the window to 1800x1000 pixels

style = ttk.Style()
style.configure('TFrame', background='#FFFFFF')  # Set the background color for ttk.Frame

# Create a frame to hold the scale, images, and button
frame = ttk.Frame(root, padding="20")
frame.grid(row=4, column=4, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

# Create a variable to store the selected percentage
percentage_var = tk.IntVar(value=0)

# Create the scale in the middle of the screen with increased length
scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal", length=800, variable=percentage_var, style="TScale")
scale.grid(row=4, column=1, padx=20, pady=20)

# Bind click events to change the scale value directly on click
scale.bind("<Button-1>", on_scale_click)  # Set value on click

# Create the button below the scale
button = tk.Button(frame, text="Submit", command=on_button_click, width=20, height=3)
button.grid(row=5, column=1, padx=20, pady=20)  # Increase padding for size

# Load and add the first image on the left
image_path1 = '1-02.jpg'
image1 = cv2.imread(image_path1)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
image1 = cv2.resize(image1, (200, 200), interpolation=cv2.INTER_LANCZOS4)

# Convert the image to a format compatible with Tkinter
image1 = Image.fromarray(image1)
photo1 = ImageTk.PhotoImage(image1)

left_image_label = tk.Label(frame, image=photo1)
left_image_label.grid(row=4, column=0, padx=20, pady=10)

# Load and add the second image on the right
image_path2 = '1-01.jpg'
image2 = cv2.imread(image_path2)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
image2 = cv2.resize(image2, (200, 200), interpolation=cv2.INTER_LANCZOS4)

# Convert the image to a format compatible with Tkinter
image2 = Image.fromarray(image2)
photo2 = ImageTk.PhotoImage(image2)

right_image_label = tk.Label(frame, image=photo2)
right_image_label.grid(row=4, column=3, padx=20, pady=10)

# Load and add the new image above the generated image
image_path3 = '1-03.jpg'  # Add the path to your new image
image3 = cv2.imread(image_path3)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
scale_factor = 1.1

# Get the original dimensions
original_height, original_width = image3.shape[:2]

# Calculate new dimensions
new_dimensions = (int(original_width * scale_factor), int(original_height * scale_factor))

# Resize the image using the new dimensions
# image3 = cv2.resize(image3, new_dimensions, interpolation=cv2.INTER_LANCZOS4)

# Convert the image to a format compatible with Tkinter
image3 = Image.fromarray(image3)
photo3 = ImageTk.PhotoImage(image3)

# Create a label to display the new image
top_image_label = tk.Label(frame, image=photo3)
top_image_label.grid(row=0, column=1)

# Create a label to display the generated image in the upper middle
generated_image_label = tk.Label(frame)
generated_image_label.grid(row=1, column=1, padx=20, pady=20, rowspan=2, columnspan=1)

# Run the application
root.mainloop()

