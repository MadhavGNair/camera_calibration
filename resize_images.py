import os

from PIL import ExifTags, Image


def correct_image_orientation(image):
    try:
        # check for EXIF orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        # if orientation exists, rotate the image
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # if no EXIF data or orientation tag, do nothing
        pass

    return image


def resize_images(input_folder, output_folder, target_size=(600, 800)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # open the image
            with Image.open(os.path.join(input_folder, filename)) as img:
                # correct orientation first
                img = correct_image_orientation(img)

                # resize using high-quality Lanczos method
                resized_img = img.resize(target_size, Image.LANCZOS)

                # save as PNG for lossless quality
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_folder, output_filename)
                resized_img.save(output_path)
                print(f"Resized: {output_filename}")


def rename_images(folder_path):
    # get all PNG files and sort them
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    png_files.sort()

    for index, filename in enumerate(png_files):
        # create new filename
        new_filename = f"e_{index}.png"

        # full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # rename
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    input_folder = "./images"
    output_folder = "./images"
    # resize_images(input_folder, output_folder)
