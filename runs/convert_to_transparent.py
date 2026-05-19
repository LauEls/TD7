from PIL import Image

def make_white_transparent(input_path, output_path):
    # Load the image
    img = Image.open(input_path)
    
    # Convert to RGBA if it isn't already (adds the alpha channel)
    img = img.convert("RGBA")
    
    datas = img.getdata()
    new_data = []

    for item in datas:
        # item is a tuple: (R, G, B, A)
        # Check if the RGB values are 255 (white)
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            # Replace with transparent (0 alpha)
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    # Update image data and save
    img.putdata(new_data)
    img.save(output_path, "PNG")
    print(f"Success! Saved to {output_path}")

# Usage
file_name = "mean_percentage_demonstrations"
make_white_transparent(f"/home/laurenz/Downloads/{file_name}.png", f"/home/laurenz/Downloads/{file_name}_transparent.png")