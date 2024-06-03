from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage import color as sk_color
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm import tqdm


def modify_transparency(img, target_rgb):
    # 画像を読み込む
    copy_img = img.copy()
    data = copy_img.getdata()

    # 新しいピクセルデータを作成
    new_data = []
    for item in data:
        # 指定されたRGB値のピクセルの場合、透明度を255に設定
        if item[:3] == target_rgb:
            new_data.append((item[0], item[1], item[2], 255))
        else:
            # それ以外の場合、透明度を0に設定
            new_data.append((item[0], item[1], item[2], 0))

    # 新しいデータを画像に設定し直す
    copy_img.putdata(new_data)
    return copy_img


def replace_color(image, color_1, color_2, alpha_np):
    # 画像データを配列に変換
    data = np.array(image)

    # RGBAモードの画像であるため、形状変更時に4チャネルを考慮
    original_shape = data.shape

    from collections import deque

    # color_1 = np.array(color_1)
    # color_2 = np.array(color_2)
    # que = deque(
    #     zip(*np.where(data[:, :, :3] == color_1)),
    # )

    data = data.reshape(-1, 4)  # RGBAのため、4チャネルでフラット化

    # for y in range(data.shape[0]):
    #     for x in range(data.shape[1]):
    #         neighbors = [
    #             (x - 1, y),
    #             (x + 1, y),
    #             (x, y - 1),
    #             (x, y + 1),  # 上下左右
    #         ]
    #         if np.all(data[y, x, :3] == color_1, axis=2) or np.all(
    #             data[y, x, :3] == color_2, axis=2
    #         ):
    #             continue
    #         for nx, ny in neighbors:
    #             if (
    #                 nx < 0
    #                 or nx >= original_shape[1]
    #                 or ny < 0
    #                 or ny >= original_shape[0]
    #             ):
    #                 continue
    #             if np.all(data[ny, nx, :3] == color_1, axis=2):
    #                 que.append((y, x))
    #                 break
    #
    # while len(que) > 0:
    #     y, x = que.popleft()
    #     neighbors = [
    #         (x - 1, y),
    #         (x + 1, y),
    #         (x, y - 1),
    #         (x, y + 1),  # 上下左右
    #     ]
    #     for nx, ny in neighbors:
    #         if nx < 0 or nx >= original_shape[1] or ny < 0 or ny >= original_shape[0]:
    #             continue
    #         if np.all(data[ny, nx, :3] == color_1):
    #             data[ny, nx, :3] = data[y, x, :3]
    #             que.append((y, x))

    # color_1のマッチングを検索する際にはRGB値のみを比較
    matches = np.all(data[:, :3] == color_1, axis=1)

    # 変更を追跡するためのフラグ
    nochange_count = 0
    idx = 0

    while np.any(matches):
        idx += 1
        new_matches = np.zeros_like(matches)
        match_num = np.sum(matches)
        for i in tqdm(range(len(data))):
            if matches[i]:
                x, y = divmod(i, original_shape[1])
                neighbors = [
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),  # 上下左右
                ]
                replacement_found = False
                for nx, ny in neighbors:
                    if 0 <= nx < original_shape[0] and 0 <= ny < original_shape[1]:
                        ni = nx * original_shape[1] + ny
                        # RGBのみ比較し、アルファは無視
                        if not np.all(data[ni, :3] == color_1, axis=0) and not np.all(
                            data[ni, :3] == color_2, axis=0
                        ):
                            data[i, :3] = data[ni, :3]  # RGB値のみ更新
                            replacement_found = True
                            continue
                if not replacement_found:
                    new_matches[i] = True
        matches = new_matches
        if match_num == np.sum(matches):
            nochange_count += 1
        if nochange_count > 5:
            break

    # 最終的な画像をPIL形式で返す
    data = data.reshape(original_shape)
    data[:, :, 3] = 255 - alpha_np
    return Image.fromarray(data, "RGBA")


def recolor_lineart_and_composite(lineart_image, base_image, new_color, alpha_th):
    """
    Recolor an RGBA lineart image to a single new color while preserving alpha, and composite it over a base image.

    Args:
    lineart_image (PIL.Image): The lineart image with RGBA channels.
    base_image (PIL.Image): The base image to composite onto.
    new_color (tuple): The new RGB color for the lineart (e.g., (255, 0, 0) for red).

    Returns:
    PIL.Image: The composited image with the recolored lineart on top.
    """
    # Ensure images are in RGBA mode
    if lineart_image.mode != "RGBA":
        lineart_image = lineart_image.convert("RGBA")
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")

    # Extract the alpha channel from the lineart image
    r, g, b, alpha = lineart_image.split()

    alpha_np = np.array(alpha)
    alpha_np[alpha_np < alpha_th] = 0
    alpha_np[alpha_np >= alpha_th] = 255

    new_alpha = Image.fromarray(alpha_np)

    # Create a new image using the new color and the alpha channel from the original lineart
    new_lineart_image = Image.merge(
        "RGBA",
        (
            Image.new("L", lineart_image.size, int(new_color[0])),
            Image.new("L", lineart_image.size, int(new_color[1])),
            Image.new("L", lineart_image.size, int(new_color[2])),
            new_alpha,
        ),
    )

    # Composite the new lineart image over the base image
    composite_image = Image.alpha_composite(base_image, new_lineart_image)

    return composite_image, alpha_np


def thicken_and_recolor_lines(base_image, lineart, thickness=3, new_color=(0, 0, 0)):
    """
    Thicken the lines of a lineart image, recolor them, and composite onto another image,
    while preserving the transparency of the original lineart.

    Args:
    base_image (PIL.Image): The base image to composite onto.
    lineart (PIL.Image): The lineart image with transparent background.
    thickness (int): The desired thickness of the lines.
    new_color (tuple): The new color to apply to the lines (R, G, B).

    Returns:
    PIL.Image: The image with the recolored and thickened lineart composited on top.
    """
    # Ensure both images are in RGBA format
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")
    if lineart.mode != "RGB":
        lineart = lineart.convert("RGBA")

    # Convert the lineart image to OpenCV format
    lineart_cv = np.array(lineart)

    white_pixels = np.sum(lineart_cv == 255)
    black_pixels = np.sum(lineart_cv == 0)

    lineart_gray = cv2.cvtColor(lineart_cv, cv2.COLOR_RGBA2GRAY)

    if white_pixels > black_pixels:
        lineart_gray = cv2.bitwise_not(lineart_gray)

    # Thicken the lines using OpenCV
    kernel = np.ones((thickness, thickness), np.uint8)
    lineart_thickened = cv2.dilate(lineart_gray, kernel, iterations=1)
    lineart_thickened = cv2.bitwise_not(lineart_thickened)
    # Create a new RGBA image for the recolored lineart
    lineart_recolored = np.zeros_like(lineart_cv)
    lineart_recolored[:, :, :3] = new_color  # Set new RGB color

    lineart_recolored[:, :, 3] = np.where(
        lineart_thickened < 250, 255, 0
    )  # Blend alpha with thickened lines

    # Convert back to PIL Image
    lineart_recolored_pil = Image.fromarray(lineart_recolored, "RGBA")

    # Composite the thickened and recolored lineart onto the base image
    combined_image = Image.alpha_composite(base_image, lineart_recolored_pil)

    return combined_image


def generate_distant_colors(consolidated_colors, distance_threshold):
    """
    Generate new RGB colors that are at least 'distance_threshold' CIEDE2000 units away from given colors.

    Args:
    consolidated_colors (list of tuples): List of ((R, G, B), count) tuples.
    distance_threshold (float): The minimum CIEDE2000 distance from the given colors.

    Returns:
    list of tuples: List of new RGB colors that meet the distance requirement.
    """
    # new_colors = []
    # Convert the consolidated colors to LAB
    consolidated_lab = [
        rgb2lab(np.array([color], dtype=np.float32) / 255.0).reshape(3)
        for color, _ in consolidated_colors
    ]

    # Try to find a distant color
    max_attempts = 10000
    for _ in range(max_attempts):
        # Generate a random color in RGB and convert to LAB
        random_rgb = np.random.randint(0, 256, size=3)
        random_lab = rgb2lab(np.array([random_rgb], dtype=np.float32) / 255.0).reshape(
            3
        )
        for base_color_lab in consolidated_lab:
            # Calculate the CIEDE2000 distance
            distance = deltaE_ciede2000(base_color_lab, random_lab)
            if distance <= distance_threshold:
                break
        new_color = tuple(random_rgb)
        break
    return new_color


def consolidate_colors(major_colors, threshold):
    """
    Consolidate similar colors in the major_colors list based on the CIEDE2000 metric.

    Args:
    major_colors (list of tuples): List of ((R, G, B), count) tuples.
    threshold (float): Threshold for CIEDE2000 color difference.

    Returns:
    list of tuples: Consolidated list of ((R, G, B), count) tuples.
    """
    # Convert RGB to LAB
    colors_lab = [
        rgb2lab(np.array([[color]], dtype=np.float32) / 255.0).reshape(3)
        for color, _ in major_colors
    ]
    n = len(colors_lab)

    # Find similar colors and consolidate
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            delta_e = deltaE_ciede2000(colors_lab[i], colors_lab[j])
            if delta_e < threshold:
                # Compare counts and consolidate to the color with the higher count
                if major_colors[i][1] >= major_colors[j][1]:
                    major_colors[i] = (
                        major_colors[i][0],
                        major_colors[i][1] + major_colors[j][1],
                    )
                    major_colors.pop(j)
                    colors_lab.pop(j)
                else:
                    major_colors[j] = (
                        major_colors[j][0],
                        major_colors[j][1] + major_colors[i][1],
                    )
                    major_colors.pop(i)
                    colors_lab.pop(i)
                n -= 1
                continue
            j += 1
        i += 1

    return major_colors


def get_major_colors(image, threshold_percentage=0.01):
    """
    Analyze an image to find the major RGB values based on a threshold percentage.

    Args:
    image (PIL.Image): The image to analyze.
    threshold_percentage (float): The percentage threshold to consider a color as major.

    Returns:
    list of tuples: A list of (color, count) tuples for colors that are more frequent than the threshold.
    """
    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Count each color
    color_count = defaultdict(int)
    for pixel in image.getdata():
        color_count[pixel] += 1

    # Total number of pixels
    total_pixels = image.width * image.height

    # Filter colors to find those above the threshold
    major_colors = [
        (color, count)
        for color, count in color_count.items()
        if (count / total_pixels) >= threshold_percentage
    ]

    return major_colors


def process(image, lineart, alpha_th, thickness):
    org = image

    major_colors = get_major_colors(image, threshold_percentage=0.05)
    major_colors = consolidate_colors(major_colors, 10)
    new_color_1 = generate_distant_colors(major_colors, 100)
    image = thicken_and_recolor_lines(
        org, lineart, thickness=thickness, new_color=new_color_1
    )
    major_colors.append((new_color_1, 0))
    new_color_2 = generate_distant_colors(major_colors, 100)
    image, alpha_np = recolor_lineart_and_composite(
        lineart, image, new_color_2, alpha_th
    )
    image = replace_color(image, new_color_1, new_color_2, alpha_np)
    unfinished = modify_transparency(image, new_color_1)

    return image, unfinished


def main():
    # parse argvs
    # python starline.py -c COLORED_IMAGE -l LINEART_IMAGE [-o OUTPUT_DIR] [-a alpha_th] [-t thickness]
    # write output_dir/RESULT_IMAGE.png

    # use argparse to parse argvs
    import os
    import sys
    from argparse import ArgumentParser

    from PIL import Image

    from utils import randomname

    args = ArgumentParser(
        prog="starline",
        description="Starline",
        epilog="Starline",
    )
    args.add_argument("-c", "--colored_image", help="colored image", required=True)
    args.add_argument("-l", "--lineart_image", help="lineart image", required=True)
    args.add_argument("-o", "--output_dir", help="output directory", default=".")
    args.add_argument("-a", "--alpha_th", help="alpha threshold", default=100)
    args.add_argument("-t", "--thickness", help="line thickness", default=5)

    sys.argv += ["-c", "color.png"]
    sys.argv += ["-l", "line.png"]

    args = args.parse_args(sys.argv[1:])
    colored_image_path = args.colored_image
    lineart_image_path = args.lineart_image
    alpha = args.alpha_th
    thickness = args.thickness
    output_dir = args.output_dir

    colored_image = Image.open(colored_image_path)
    lineart_image = Image.open(lineart_image_path).convert("RGBA")

    result_image, unfinished = process(colored_image, lineart_image, alpha, thickness)

    output_image = Image.alpha_composite(result_image, lineart_image)

    name = randomname(10)

    os.makedirs(f"{output_dir}/{name}")
    output_image.save(f"{output_dir}/{name}/output_image.png")
    result_image.save(f"{output_dir}/{name}/color_image.png")
    unfinished.save(f"{output_dir}/{name}/unfinished_image.png")


if __name__ == "__main__":
    main()
