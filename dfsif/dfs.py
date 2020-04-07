import random

import numpy as np


def depth_first_search_filter(image, cell_size, border_width, pooling, border_pooling):
    """
    Generates a depth-first search filtered image.

    :param image: image to be filtered
    :param cell_size: length of the cells the image will be divided into
    :param border_width: border width inside each cell, effectively half of final border width
    :param pooling: function to pool the color values covered by a given cell
    :param border_pooling: function to pool the color values covered by a given cell for the cell border
    :return: depth-first search filtered image
    """
    height = image.shape[0] // cell_size
    width = image.shape[1] // cell_size

    visited = np.zeros((height, width), dtype=np.bool)

    borders = np.ones(visited.shape + (4,), dtype=np.bool)

    filtered = np.zeros_like(image)[:height * cell_size, :width * cell_size]

    open_list = []

    up = np.array([1, 0])
    right = np.array([0, 1])

    def is_valid(c):
        return 0 <= c[0] < height and 0 <= c[1] < width

    def neighbors_of(c):
        """Returns valid neighboring cells that have not yet been visited."""
        neighbors = [c + up, c - up, c + right, c - right]
        neighbors = [(n, c) for n in neighbors if is_valid(n) and not visited[n[0], n[1]]]
        random.shuffle(neighbors)
        return neighbors

    origin = np.array([random.randrange(height), random.randrange(width)])
    visited[origin[0], origin[1]] = True
    open_list.extend(neighbors_of(origin))

    while len(open_list) > 0:
        cell, parent = open_list.pop()
        if visited[cell[0], cell[1]]:
            continue
        visited[cell[0], cell[1]] = True
        north = np.any(cell != parent + up)
        south = np.any(cell != parent - up)
        east = np.any(cell != parent + right)
        west = np.any(cell != parent - right)
        borders[parent[0], parent[1]] &= [north, south, east, west]
        borders[cell[0], cell[1]] &= [south, north, west, east]

        open_list.extend(neighbors_of(cell))

    for y in range(height):
        for x in range(width):
            start_y = y * cell_size
            end_y = start_y + cell_size
            start_x = x * cell_size
            end_x = start_x + cell_size

            # Get colors
            color = pooling(image[start_y:end_y, start_x:end_x])
            border_color = border_pooling(image[start_y:end_y, start_x:end_x])

            # Draw border corners
            filtered[start_y:end_y, start_x:end_x] = color
            filtered[start_y:start_y + border_width, start_x:start_x + border_width] = border_color
            filtered[start_y:start_y + border_width, end_x - border_width:end_x] = border_color
            filtered[end_y - border_width:end_y, start_x:start_x + border_width] = border_color
            filtered[end_y - border_width:end_y, end_x - border_width:end_x] = border_color

            # Draw borders
            north, south, east, west = borders[y, x]
            if north:
                filtered[end_y - border_width:end_y, start_x:end_x] = border_color
            if south:
                filtered[start_y:start_y + border_width, start_x:end_x] = border_color
            if east:
                filtered[start_y:end_y, end_x - border_width:end_x] = border_color
            if west:
                filtered[start_y:end_y, start_x:start_x + border_width] = border_color

    return filtered
