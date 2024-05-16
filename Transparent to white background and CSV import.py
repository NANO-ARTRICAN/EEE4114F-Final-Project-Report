from PIL import Image
import csv
import os

def save_swapped_binary_pixels_to_csv(input_folders, csv_path):
    # Prepare header row with pixel numbers
    pixel_numbers = ["Pixel" + str(i) for i in range(1, 1 + 3 * 256 * 256)]
    
    # Write header row to CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Folder'] + pixel_numbers)  # Write 'Folder' in the first cell of the header row
    
    # Iterate over input folders
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):  # Assuming images are PNG or JPEG format
                    image_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)
                    save_swapped_binary_pixel_to_csv_single(image_path, csv_path, folder_name)

    print("Swapped binary pixel values for all images saved to", csv_path)

def save_swapped_binary_pixel_to_csv_single(image_path, csv_path, folder_name):
    # Open the image
    img = Image.open(image_path)
    
    # Get pixel values
    pixels = list(img.getdata())
    
    # Convert pixel values over 100 to 0 and others to 1
    binary_pixels = [0 if sum(pixel) // 3 > 100 else 1 for pixel in pixels]
    
    # Append binary pixel values to CSV file
    with open(csv_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([folder_name] + binary_pixels)

directory = "C:/Users/tko20/Music/2024 SEM1/EEEE4114F/Project/Databank - Copy/"
input_folders = [directory+str(0),directory+str(1),directory+str(2),directory+str(3),directory+str(4),directory+str(5),directory+str(6),directory+str(7),directory+str(8),directory+str(9) ]
output_csv_path = "pixel_values.csv"  # Provide the output path for the CSV file
save_swapped_binary_pixels_to_csv(input_folders, output_csv_path)
#still have to manually remove the extra pixel number headers