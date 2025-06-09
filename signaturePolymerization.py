import os
from PIL import Image
from datasetsCreation import resizeSignatures
def combineMatchingImages(folder11, folder12,folder21, folder22, output_folder):
    """
    Combines images from folder1 and folder2 side by side,
    if their filenames (without the last character) match.
    Saves the combined image in the output folder.
    """

    os.makedirs(output_folder, exist_ok=True)
    resizeSignatures(folder11,folder12,512,512)
    resizeSignatures(folder21,folder22,512,512)
    spacing = 20
    def get_image_map(folder):
        image_map = {}
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                base = os.path.splitext(fname)[0][:-1]  # remove last character
                image_map[base] = os.path.join(folder, fname)
        return image_map

    map1 = get_image_map(folder12)
    map2 = get_image_map(folder22)

    matched = 0

    for key in map1:
        if key in map2:
            img1 = Image.open(map1[key])
            img2 = Image.open(map2[key])

            # Resize to same height if needed
            if img1.size[1] != img2.size[1]:
                new_height = min(img1.size[1], img2.size[1])
                img1 = img1.resize((int(img1.size[0] * new_height / img1.size[1]), new_height))
                img2 = img2.resize((int(img2.size[0] * new_height / img2.size[1]), new_height))

            # Calculate total width including spacing
            total_width = img1.size[0] + spacing + img2.size[0]
            combined_img = Image.new('RGB', (total_width, img1.size[1]), color=(255, 255, 255))  # white background

            # Paste images with spacing in between
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (img1.size[0] + spacing, 0))

            output_name = f"{key}_combined.jpg"
            combined_img.save(os.path.join(output_folder, output_name))
            matched += 1
            print(f"✅ Combined with spacing: {output_name}")

    if matched == 0:
        print("⚠️ No matching image pairs found.")

