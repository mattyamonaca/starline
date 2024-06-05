from collections import defaultdict, deque

import cv2
import numpy as np
from PIL import Image
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

    color_1 = np.array(color_1, dtype=np.uint8)
    color_2 = np.array(color_2, dtype=np.uint8)

    # 幅優先探索で color_1 の領域を外側から塗りつぶす
    # color_2 で保護されたオリジナルの線画
    protected = np.all(data[:, :, :3] == color_2, axis=2)
    # color_1 で塗られた塗りつぶしたい領域
    fill_target = np.all(data[:, :, :3] == color_1, axis=2)
    # すでに塗られている領域
    colored = (protected | fill_target) == False

    # bfs の始点を列挙
    # colored をそのまま使ってもいいが、pythonは遅いのでnumpy経由のこの方が速い
    # 上下左右にシフトした fill_target & colored == True になるやつ
    adj_r = colored & np.roll(fill_target, -1, axis=0)
    adj_r[:, -1] = False
    adj_l = colored & np.roll(fill_target, 1, axis=0)
    adj_l[:, 0] = False
    adj_u = colored & np.roll(fill_target, 1, axis=1)
    adj_u[:, 0] = False
    adj_d = colored & np.roll(fill_target, -1, axis=1)
    adj_d[:, -1] = False

    # そのピクセルはすでに塗られていて、上下左右いずれかのピクセルが color_1 であるもの
    bfs_start = adj_r | adj_l | adj_u | adj_d

    que = deque(
        zip(*np.where(bfs_start)),
        maxlen=original_shape[0] * original_shape[1] * 2,
    )

    with tqdm(total=original_shape[0] * original_shape[1]) as pbar:
        pbar.update(np.sum(colored) - np.sum(bfs_start) + np.sum(protected))
        while len(que) > 0:
            y, x = que.popleft()
            neighbors = [
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),  # 上下左右
            ]
            pbar.update(1)
            # assert not fill_target[y, x] and not protected[y, x]
            # assert colored[y, x]
            color = data[y, x, :3]

            for nx, ny in neighbors:
                if (
                    nx < 0
                    or nx >= original_shape[1]
                    or ny < 0
                    or ny >= original_shape[0]
                ):
                    continue
                if fill_target[ny, nx]:
                    fill_target[ny, nx] = False
                    # colored[ny, nx] = True
                    data[ny, nx, :3] = color
                    que.append((ny, nx))
        pbar.update(pbar.total - pbar.n)

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
    max_attempts = 1000
    best_dist = 0.0
    best_color = (0, 0, 0)

    # np.random.seed(42)
    for _ in range(max_attempts):
        # Generate a random color in RGB and convert to LAB
        random_rgb = np.random.randint(0, 256, size=3)
        random_lab = rgb2lab(np.array([random_rgb], dtype=np.float32) / 255.0).reshape(
            3
        )
        # consolidated_lab にある色からできるだけ遠い色を選びたい
        min_distance = min(
            map(
                lambda base_color_lab: deltaE_ciede2000(base_color_lab, random_lab),
                consolidated_lab,
            )
        )
        if min_distance > distance_threshold:
            return tuple(random_rgb)
        # 閾値以上のものが見つからなかった場合に備えて一番良かったものを覚えておく
        if best_dist < min_distance:
            best_dist = min_distance
            best_color = tuple(random_rgb)
    return best_color


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
    image.save("tmp.png")

    major_colors = get_major_colors(image, threshold_percentage=0.05)
    major_colors = consolidate_colors(major_colors, 10)

    th = 10
    threshold_percentage = 0.05
    while len(major_colors) < 1:
        threshold_percentage = threshold_percentage - 0.001
        major_colors = get_major_colors(image, threshold_percentage=threshold_percentage)
        
    while len(major_colors) < 1:
        th = th + 1
        major_colors = consolidate_colors(major_colors, th)

    new_color_1 = generate_distant_colors(major_colors, 50)
    image = thicken_and_recolor_lines(
        org, lineart, thickness=thickness, new_color=new_color_1
    )
    
    major_colors.append((new_color_1, 0))
    new_color_2 = generate_distant_colors(major_colors, 40)
    image, alpha_np = recolor_lineart_and_composite(
        lineart, image, new_color_2, alpha_th
    )
    # import time
    # start = time.time()
    image = replace_color(image, new_color_1, new_color_2, alpha_np)
    # end = time.time()
    # print(f"{end-start} sec")
    unfinished = modify_transparency(image, new_color_1)

    return image, unfinished


def main():
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
    args.add_argument("-o", "--output_dir", help="output directory", default="output")
    args.add_argument("-a", "--alpha_th", help="alpha threshold", default=100, type=int)
    args.add_argument("-t", "--thickness", help="line thickness", default=5, type=int)

    args = args.parse_args(sys.argv[1:])
    colored_image_path = args.colored_image
    lineart_image_path = args.lineart_image
    alpha = args.alpha_th
    thickness = args.thickness
    output_dir = args.output_dir

    colored_image = Image.open(colored_image_path)
    lineart_image = Image.open(lineart_image_path)
    if lineart_image.mode == "P" or lineart_image.mode == "L":
        # 線画が 1-channel 画像のときの処理
        # alpha-channel の情報が入力されたと仮定して (透明 -> 0, 不透明 -> 255)
        # RGB channel はこれを反転させたものにする (透明 -> 白 -> 255, 不透明 -> 黒 -> 0)
        lineart_image = lineart_image.convert("RGBA")
        lineart_image = np.array(lineart_image)
        lineart_image[:, :, 0] = 255 - lineart_image[:, :, 3]
        lineart_image[:, :, 1] = 255 - lineart_image[:, :, 3]
        lineart_image[:, :, 2] = 255 - lineart_image[:, :, 3]
        lineart_image = Image.fromarray(lineart_image)
        lineart_image = lineart_image.convert("RGBA")

    result_image, unfinished = process(colored_image, lineart_image, alpha, thickness)

    output_image = Image.alpha_composite(result_image, lineart_image)

    name = randomname(10)

    os.makedirs(f"{output_dir}/{name}")
    output_image.save(f"{output_dir}/{name}/output_image.png")
    result_image.save(f"{output_dir}/{name}/color_image.png")
    unfinished.save(f"{output_dir}/{name}/unfinished_image.png")


if __name__ == "__main__":
    main()
