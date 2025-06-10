from PIL import Image

def merge_images(image_paths, output_path):
    # Ensure there are exactly 9 images
    if len(image_paths) != 9:
        raise ValueError("Exactly 9 images are required to merge.")

    # Open all images and ensure they are full size
    images = [Image.open(img_path) for img_path in image_paths]

    # Get the size of a single image (assuming all are the same size)
    img_width, img_height = images[0].size

    # Create a blank canvas for the 3x3 grid
    grid_width = img_width * 3
    grid_height = img_height * 3
    merged_image = Image.new('RGB', (grid_width, grid_height))

    # Paste images into the grid
    for idx, img in enumerate(images):
        # Ensure the image is not resized
        if img.size != (img_width, img_height):
            raise ValueError(f"Image {image_paths[idx]} has a different size: {img.size}. All images must have the same dimensions.")

        x_offset = (idx % 3) * img_width
        y_offset = (idx // 3) * img_height
        merged_image.paste(img, (x_offset, y_offset))

    # Save the merged image
    merged_image.save(output_path)
    print(f"Merged image saved to {output_path}")


# Example usage
image_files = [
    "./figures/bouldering_ch3_outliers/C0.5_fig1.png", "./figures/bouldering_ch3_outliers/C0.5_fig2.png", "./figures/bouldering_ch3_outliers/C0.5_fig3.png",
    "./figures/bouldering_ch3_outliers/C1_fig1.png", "./figures/bouldering_ch3_outliers/C1_fig2.png", "./figures/bouldering_ch3_outliers/C1_fig3.png",
    "./figures/bouldering_ch3_outliers/C2_fig1.png", "./figures/bouldering_ch3_outliers/C2_fig2.png", "./figures/bouldering_ch3_outliers/C2_fig3.png"
]
merge_images(image_files, "merged_image_Chauvenet.png")