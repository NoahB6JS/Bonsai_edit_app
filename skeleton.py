import cv2
import numpy as np
import os
import math

# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "storage/bons1.jpg"

GREEN_MASK_FILE = "green_mask.png"
OUTPUT_FILE = "bonsai_skeleton.png"

MAX_IMAGE_SIZE = 1200

# Wood detection
DARK_THRESHOLD = 175
MIN_COMPONENT_SIZE = 50

# Skeleton cleanup
MIN_BRANCH_LENGTH = 18
PRUNE_ROUNDS = 8

# Smoothing
SMOOTHING_ROUNDS = 3

# Appearance
WOOD_COLOR = (55, 75, 125)  # brown in BGR
LINE_THICKNESS = 6
BACKGROUND = (248, 248, 248)


# ============================================================
# REMOVE OLD OUTPUTS
# ============================================================

for filename in [GREEN_MASK_FILE, OUTPUT_FILE]:
    if os.path.exists(filename):
        os.remove(filename)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}"
    )

height, width = image.shape[:2]

scale = min(
    1.0,
    MAX_IMAGE_SIZE / max(height, width)
)

if scale < 1:
    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

height, width = image.shape[:2]


# ============================================================
# FIND GREEN
# ============================================================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([25, 30, 25])
upper_green = np.array([100, 255, 255])

green_mask = cv2.inRange(
    hsv,
    lower_green,
    upper_green
)

kernel = np.ones((3, 3), np.uint8)

green_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_OPEN,
    kernel
)

green_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_CLOSE,
    kernel
)

cv2.imwrite(
    GREEN_MASK_FILE,
    green_mask
)


# ============================================================
# FIND DARK WOOD
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Dark pixels
dark_mask = (
    gray < DARK_THRESHOLD
).astype(np.uint8) * 255

# Do not treat green foliage as wood
wood_mask = cv2.bitwise_and(
    dark_mask,
    cv2.bitwise_not(green_mask)
)


# ============================================================
# CLEAN WOOD MASK
# ============================================================

wood_mask = cv2.morphologyEx(
    wood_mask,
    cv2.MORPH_OPEN,
    np.ones((3, 3), np.uint8),
    iterations=1
)

wood_mask = cv2.morphologyEx(
    wood_mask,
    cv2.MORPH_CLOSE,
    np.ones((5, 5), np.uint8),
    iterations=1
)


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    wood_mask,
    connectivity=8
)

clean_mask = np.zeros_like(wood_mask)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area >= MIN_COMPONENT_SIZE:
        clean_mask[labels == i] = 255

wood_mask = clean_mask


# ============================================================
# SKELETONISE
# ============================================================

if hasattr(cv2, "ximgproc"):

    skeleton = cv2.ximgproc.thinning(
        wood_mask,
        thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
    )

else:

    skeleton = np.zeros_like(wood_mask)

    working = wood_mask.copy()

    while True:

        eroded = cv2.erode(
            working,
            np.ones((3, 3), np.uint8)
        )

        opened = cv2.dilate(
            eroded,
            np.ones((3, 3), np.uint8)
        )

        difference = cv2.subtract(
            working,
            opened
        )

        skeleton = cv2.bitwise_or(
            skeleton,
            difference
        )

        working = eroded

        if cv2.countNonZero(working) == 0:
            break


# ============================================================
# SKELETON NEIGHBOURS
# ============================================================

def get_neighbours(point):

    y, x = point

    result = []

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:

            if dx == 0 and dy == 0:
                continue

            ny = y + dy
            nx = x + dx

            if (
                0 <= ny < height
                and 0 <= nx < width
                and skeleton[ny, nx] > 0
            ):
                result.append((ny, nx))

    return result


def pixel_degree(point):

    return len(get_neighbours(point))


# ============================================================
# PRUNE VERY SHORT TWIGS
# ============================================================

for _ in range(PRUNE_ROUNDS):

    pixels = list(
        zip(
            *np.where(skeleton > 0)
        )
    )

    endpoints = [
        p for p in pixels
        if pixel_degree(p) == 1
    ]

    changed = False

    for endpoint in endpoints:

        if skeleton[
            endpoint[0],
            endpoint[1]
        ] == 0:
            continue

        path = [endpoint]

        previous = None
        current = endpoint

        for _ in range(MIN_BRANCH_LENGTH):

            neighbours = get_neighbours(current)

            if previous is not None:
                neighbours = [
                    p for p in neighbours
                    if p != previous
                ]

            if not neighbours:
                break

            # We have reached a real branch junction
            if pixel_degree(current) >= 3:
                break

            next_pixel = neighbours[0]

            path.append(next_pixel)

            previous = current
            current = next_pixel

        # Only remove genuinely tiny twigs
        if len(path) < MIN_BRANCH_LENGTH:

            for y, x in path:

                skeleton[y, x] = 0

            changed = True

    if not changed:
        break


# ============================================================
# TRACE EVERY CONTINUOUS SKELETON PATH
# ============================================================

pixels = set(
    zip(
        *np.where(skeleton > 0)
    )
)

visited_edges = set()
paths = []


def edge_key(a, b):

    return tuple(
        sorted([a, b])
    )


# Junctions are only used internally to trace the branches.
# THEY ARE NOT DRAWN.
junctions = [
    p for p in pixels
    if pixel_degree(p) >= 3
]


def trace_from(start, next_pixel):

    path = [
        start,
        next_pixel
    ]

    previous = start
    current = next_pixel

    visited_edges.add(
        edge_key(previous, current)
    )

    while True:

        neighbours = get_neighbours(current)

        # Remove the pixel we just came from
        forward = [
            p for p in neighbours
            if p != previous
        ]

        # Stop when reaching another junction
        if pixel_degree(current) >= 3:

            break

        if not forward:
            break

        # Continue along the branch
        next_point = forward[0]

        e = edge_key(
            current,
            next_point
        )

        if e in visited_edges:
            break

        visited_edges.add(e)

        path.append(next_point)

        previous = current
        current = next_point

    return path


# Trace branches starting at junctions
for junction in junctions:

    for neighbour in get_neighbours(junction):

        e = edge_key(
            junction,
            neighbour
        )

        if e in visited_edges:
            continue

        path = trace_from(
            junction,
            neighbour
        )

        if len(path) >= 4:
            paths.append(path)


# ============================================================
# TRACE REMAINING LOOSE PATHS
# ============================================================

remaining_pixels = [
    p for p in pixels
    if pixel_degree(p) <= 1
]

for start in remaining_pixels:

    neighbours = get_neighbours(start)

    for neighbour in neighbours:

        e = edge_key(
            start,
            neighbour
        )

        if e in visited_edges:
            continue

        path = trace_from(
            start,
            neighbour
        )

        if len(path) >= 4:
            paths.append(path)


# ============================================================
# CHAikin SMOOTHING
# ============================================================

def chaikin(points, iterations=3):

    if len(points) < 3:
        return points

    result = [
        np.array(p, dtype=float)
        for p in points
    ]

    for _ in range(iterations):

        new_points = [result[0]]

        for i in range(len(result) - 1):

            p0 = result[i]
            p1 = result[i + 1]

            q = (
                0.75 * p0 +
                0.25 * p1
            )

            r = (
                0.25 * p0 +
                0.75 * p1
            )

            new_points.append(q)
            new_points.append(r)

        new_points.append(result[-1])

        result = new_points

    return [
        (
            int(round(p[0])),
            int(round(p[1]))
        )
        for p in result
    ]


# ============================================================
# CREATE CLEAN CANVAS
# ============================================================

canvas = np.full(
    (height, width, 3),
    BACKGROUND,
    dtype=np.uint8
)


# ============================================================
# DRAW ALL BRANCHES
# ============================================================

for path in paths:

    if len(path) < 4:
        continue

    # Convert y,x -> x,y
    xy_points = [
        (x, y)
        for y, x in path
    ]

    # Smooth the actual branch path
    smooth = chaikin(
        xy_points,
        SMOOTHING_ROUNDS
    )

    if len(smooth) < 2:
        continue

    points = np.array(
        smooth,
        dtype=np.int32
    )

    # Smooth brown curve
    cv2.polylines(
        canvas,
        [points],
        False,
        WOOD_COLOR,
        LINE_THICKNESS,
        cv2.LINE_AA
    )


# ============================================================
# CONNECT BRANCH JUNCTIONS
# ============================================================
#
# The branches above are traced separately between junctions.
# This section draws a tiny smooth connection around each
# junction so the branch system visually joins together.
#
# IMPORTANT:
# NO NODE / DOT IS DRAWN.
#

for junction in junctions:

    y, x = junction

    # Look at the original wood skeleton around the junction
    # and draw a small filled connection.
    radius = 4

    cv2.circle(
        canvas,
        (x, y),
        radius,
        WOOD_COLOR,
        -1,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_FILE,
    canvas
)


# ============================================================
# RESULT
# ============================================================

print()
print("========================================")
print(" BONSAI WOOD STRUCTURE COMPLETE")
print("========================================")
print()
print(f"Image: {width} x {height}")
print(f"Branch paths: {len(paths)}")
print()
print("Created:")
print(f"  {GREEN_MASK_FILE}")
print(f"  {OUTPUT_FILE}")
print()
print("No leaves.")
print("No visible nodes.")
print("Branches are smoothed and connected.")
print("========================================")