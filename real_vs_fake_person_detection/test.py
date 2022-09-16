from pathlib import Path
from typing import Union
import argparse

import cv2
import numpy


def is_cartoon(
    image: Union[str, Path],
    threshold: float = 0.3,
    preview: bool = False,
) -> bool:
    # read and resize image
    img = cv2.imread(str(image))
    img = cv2.resize(img, (1024, 1024))

    # Find count of each color
    a = {}
    for row in img:
        for item in row:
            value = tuple(item)
            if value not in a:
                a[value] = 1
            else:
                a[value] += 1

    if preview:
        mask = numpy.zeros(img.shape[:2], dtype=bool)

        for color, _ in sorted(a.items(), key=lambda pair: pair[1], reverse=True)[:512]:
            mask |= (img == color).all(-1)

        img[~mask] = (255, 255, 255)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Identify the percent of the image that uses the top 512 colors
    most_common_colors = sum(
        [x[1]
            for x in sorted(a.items(), key=lambda pair: pair[1], reverse=True)[:512]]
    )
    return (most_common_colors / (1024 * 1024)) > threshold


# def command_line_options():
#     args = argparse.ArgumentParser(
#         "blur_compare",
#         description="Determine if a image is likely a cartoon or photo.",
#     )
#     args.add_argument(
#         "-p",
#         "--preview",
#         action="store_true",
#         help="Show the blurred image",
#     )
#     args.add_argument(
#         "-t",
#         "--threshold",
#         type=float,
#         help="Cutoff threshold",
#         default=0.3,
#     )
#     args.add_argument(
#         "image",
#         type=Path,
#         help="Path to image file",
#     )
#     return vars(args.parse_args())


# if __name__ == "__main__":
#     options = command_line_options()
#     if not options["image"].exists():
#         raise FileNotFoundError(
#             f"No image exists at {options['image'].absolute()}")
#     if is_cartoon(**options):
#         print(f"{options['image'].name} is a cartoon!")
#     else:
#         print(f"{options['image'].name} is a photo!")

print(is_cartoon('a5.jpg'))
