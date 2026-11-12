# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Seeded dungeon generator: a few rooms joined by corridors, plus the stairs.
# The same random generator state always builds the same map.

from collections import deque
from dataclasses import dataclass

import numpy as np

# tile ids stored in the grid
WALL = 0
FLOOR = 1
STAIRS = 2


@dataclass
class Dungeon:
    """one floor: the tile grid, where the player starts, and where the stairs are"""
    grid: np.ndarray            # (height, width) of tile ids
    start: tuple[int, int]      # (row, col)
    stairs: tuple[int, int]     # (row, col)


def generate_dungeon(rng, width=12, height=12, max_rooms=4):
    """build a dungeon using a numpy random generator"""
    rooms = []
    while len(rooms) < 2:
        rooms = place_rooms(rng, width, height, max_rooms)

    grid = np.full((height, width), WALL, dtype=np.int8)
    for top, left, room_h, room_w in rooms:
        grid[top:top + room_h, left:left + room_w] = FLOOR
    for room_a, room_b in zip(rooms, rooms[1:]):
        carve_corridor(grid, room_center(room_a), room_center(room_b), rng)

    start = random_tile_in(rooms[0], rng)
    stairs = pick_stairs(grid, start, rng)
    grid[stairs] = STAIRS
    return Dungeon(grid=grid, start=start, stairs=stairs)


def place_rooms(rng, width, height, max_rooms, tries=50):
    """randomly place up to max_rooms rooms as (top, left, height, width)"""
    rooms = []
    for _ in range(tries):
        room_h = int(rng.integers(3, 6))
        room_w = int(rng.integers(3, 6))
        top = int(rng.integers(1, height - room_h))
        left = int(rng.integers(1, width - room_w))
        new_room = (top, left, room_h, room_w)
        if not any(rooms_overlap(new_room, room) for room in rooms):
            rooms.append(new_room)
        if len(rooms) == max_rooms:
            break
    return rooms


def rooms_overlap(room_a, room_b):
    """true if two rooms touch or overlap (rooms need a 1 tile gap)"""
    top_a, left_a, h_a, w_a = room_a
    top_b, left_b, h_b, w_b = room_b
    apart = (top_a + h_a + 1 <= top_b or top_b + h_b + 1 <= top_a or
             left_a + w_a + 1 <= left_b or left_b + w_b + 1 <= left_a)
    return not apart


def room_center(room):
    """middle tile of a room"""
    top, left, room_h, room_w = room
    return top + room_h // 2, left + room_w // 2


def random_tile_in(room, rng):
    """random tile inside a room"""
    top, left, room_h, room_w = room
    return top + int(rng.integers(room_h)), left + int(rng.integers(room_w))


def carve_corridor(grid, start, end, rng):
    """dig an L shaped corridor, going sideways or up/down first at random"""
    (row1, col1), (row2, col2) = start, end
    if rng.random() < 0.5:
        grid[row1, min(col1, col2):max(col1, col2) + 1] = FLOOR
        grid[min(row1, row2):max(row1, row2) + 1, col2] = FLOOR
    else:
        grid[min(row1, row2):max(row1, row2) + 1, col1] = FLOOR
        grid[row2, min(col1, col2):max(col1, col2) + 1] = FLOOR


def distances_from(grid, start):
    """walking distance from start to every reachable tile (bfs)"""
    dist = {start: 0}
    frontier = deque([start])
    while frontier:
        row, col = frontier.popleft()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + d_row, col + d_col)
            if grid[neighbor] != WALL and neighbor not in dist:
                dist[neighbor] = dist[(row, col)] + 1
                frontier.append(neighbor)
    return dist


def pick_stairs(grid, start, rng):
    """random tile at least half as far from the start as the farthest tile"""
    dist = distances_from(grid, start)
    farthest = max(dist.values())
    options = sorted(tile for tile, d in dist.items() if d >= max(1, farthest // 2))
    return options[int(rng.integers(len(options)))]
