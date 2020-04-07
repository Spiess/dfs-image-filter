import argparse

import numpy as np
from PIL import Image

from dfsif.dfs import depth_first_search_filter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('image', help='Path to image file.', type=str)
    parser.add_argument('--cell-size', help='Cell size for search space grid.', type=int)
    parser.add_argument('--border-width', help='Border width inside each cell, effectively half of final border width.',
                        type=int, default=1)

    args = parser.parse_args()

    img = Image.open(args.image)
    img.load()
    data = np.asarray(img, dtype=np.uint8)

    width = data.shape[1]

    cell_size = args.cell_size if args.cell_size else width // 20

    result = depth_first_search_filter(data, cell_size, args.border_width,
                                       lambda x: np.mean(x, axis=(0, 1)),
                                       lambda x: np.mean(x, axis=(0, 1)) / 2)

    nimg = Image.fromarray(result)
    nimg.show()


if __name__ == '__main__':
    main()
