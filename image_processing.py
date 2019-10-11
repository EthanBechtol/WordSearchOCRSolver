import itertools
import os
import random

import cv2
import numpy as np


def resize_image(inFile: str):
    """Resize the original given image to ensure the image proportions are closer to what is expected by the program"""
    src = cv2.imread(inFile)
    dst = os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png")
    dsize = (1_286, 1_116)
    new_img = cv2.resize(src, dsize)
    cv2.imwrite(dst, new_img)


def get_canvas_forward(word: str):  # left to right
    """Create & return an image containing the given word on a blank canvas to be used for matching"""
    font = cv2.FONT_HERSHEY_DUPLEX

    word = str(word)
    im_width = len(word) * 72 - 28
    canvas = 255 * np.ones(shape=[50, im_width, 3], dtype=np.uint8)

    curr_x_pos = 0
    for letter in word.upper():
        if letter == 'I':
            cv2.putText(canvas, letter, (curr_x_pos + 13, 45), font, 2, (0, 0, 0), 2, cv2.LINE_4)

        else:
            cv2.putText(canvas, letter, (curr_x_pos, 45), font, 2, (0, 0, 0), 2, cv2.LINE_4)
        curr_x_pos += 72

    return canvas


def get_canvas_backward(word: str):  # right to left
    """Create & return an image containing the given word reversed on a blank canvas"""
    return get_canvas_forward(word[::-1])


def get_canvas_down(word: str):  # top to bottom
    """Create & return an image containing the given word on a blank canvas spanning from top to bottom"""
    letter_spacing = 36
    letter_height = 38
    padding = 18
    font = cv2.FONT_HERSHEY_DUPLEX

    word = str(word.upper())
    im_height = len(word) * (letter_height + letter_spacing) - letter_spacing + padding * 2
    canvas = 255 * np.ones(shape=[im_height, 55, 3], dtype=np.uint8)

    current_position = padding + letter_height
    for letter in word:
        if letter == 'I':
            cv2.putText(canvas, letter, (14, current_position), font, 2, (0, 0, 0), 2, cv2.LINE_4)
        else:
            cv2.putText(canvas, letter, (2, current_position), font, 2, (0, 0, 0), 2, cv2.LINE_4)
        current_position += letter_height + letter_spacing

    return canvas


def get_canvas_up(word: str):  # bottom to top
    """Create & return an image containing the given word on a blank canvas spanning from bottom to top"""
    return get_canvas_down(word[::-1])


def get_canvas_backslash(word: str):  # top-left to bottom-right
    """
    Create & return an image containing the given word on a blank canvas
    spanning from the top-left corner to bottom-right corner
    """
    letter_height_spacing = 36
    letter_height = 38
    padding_height = 18

    letter_width = 42
    font = cv2.FONT_HERSHEY_DUPLEX

    word = str(word.upper())
    im_height = len(word) * (letter_height + letter_height_spacing) - letter_height_spacing + padding_height * 2
    im_width = len(word) * 72 - 24
    canvas = 255 * np.zeros(shape=[im_height, im_width, 4], dtype=np.uint8)

    current_position_y = padding_height + letter_height
    current_position_x = 0
    for letter in word:
        if letter == '$':
            cv2.putText(canvas, letter, (current_position_x, current_position_y), font, 2, (0, 0, 0, 255), 2,
                        cv2.LINE_4)
        else:
            rect_pt1 = (current_position_x - 13, current_position_y + 20)
            rect_pt2 = (current_position_x + letter_width + 13, current_position_y - letter_height - 20)
            cv2.rectangle(canvas, rect_pt1, rect_pt2, (255, 255, 255, 255), -1)
            cv2.putText(canvas, letter, (current_position_x, current_position_y), font, 2, (0, 0, 0, 255), 2,
                        cv2.LINE_4)
        current_position_y += letter_height + letter_height_spacing
        current_position_x += 72

    cv2.imwrite("backslash_test.png", canvas)
    return canvas


def get_canvas_rbackslash(word: str):  # bottom-right to top-left
    """
    Create & return an image containing the given word on a blank canvas
    spanning from the bottom-right corner to the top-left corner
    """
    return get_canvas_backslash(word[::-1])


def get_canvas_forwardslash(word: str):  # bottom-left to top-right
    """
    Create & return an image containing the given word on a blank canvas
    spanning from the bottom-left corner to top-right corner
    """
    letter_height_spacing = 36
    letter_height = 38
    padding_height = 18

    letter_width = 42
    font = cv2.FONT_HERSHEY_DUPLEX

    word = str(word.upper())
    im_height = len(word) * (letter_height + letter_height_spacing) - letter_height_spacing + padding_height * 2
    im_width = len(word) * 72 - 24
    canvas = 255 * np.zeros(shape=[im_height, im_width, 4], dtype=np.uint8)

    current_position_y = im_height - (padding_height + letter_height) + letter_height
    current_position_x = 0
    for letter in word:
        if letter == '$':
            cv2.putText(canvas, letter, (current_position_x, current_position_y), font, 2, (0, 0, 0, 255), 2,
                        cv2.LINE_4)
        else:
            rect_pt1 = (current_position_x - 13, current_position_y + 20)
            rect_pt2 = (current_position_x + letter_width + 13, current_position_y - letter_height - 20)
            cv2.rectangle(canvas, rect_pt1, rect_pt2, (255, 255, 255, 255), -1)
            cv2.putText(canvas, letter, (current_position_x, current_position_y), font, 2, (0, 0, 0, 255), 2,
                        cv2.LINE_4)
        current_position_y -= letter_height + letter_height_spacing
        current_position_x += 72

    cv2.imwrite("backslash_test.png", canvas)
    return canvas


def get_canvas_rforwardslash(word: str):  # top-right to bottom-left
    """
    Create & return an image containing the given word on a blank canvas
    spanning from the top-right corner to the bottom-left corner
    """
    return get_canvas_forwardslash(word[::-1])


def generate_all_canvases(word):
    """
    Create & store images containing all possible orientations of the given word.
    Used in the case that Tesseract may have misread letters causing traditional array based word searching to fail.
    """
    canvas = get_canvas_forward(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_forward.png'), canvas)

    canvas = get_canvas_backward(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_backward.png'), canvas)

    canvas = get_canvas_down(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_down.png'), canvas)

    canvas = get_canvas_up(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_up.png'), canvas)

    canvas = get_canvas_backslash(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_backslash.png'), canvas)

    canvas = get_canvas_rbackslash(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_rbackslash.png'), canvas)

    canvas = get_canvas_forwardslash(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_forwardslash.png'), canvas)

    canvas = get_canvas_rforwardslash(word)
    cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_rforwardslash.png'), canvas)


def prepare_word_images(words: dict):
    """
    Create images containing the given dict of word:orientation in their proper orientations to be used
    to match them in the larger puzzle. Uses helper functions to generate the images for readability/modularity purposes
    """
    # TODO account for different spacing for larger letters (A, H, G, etc) and smaller letters (I, J, etc)
    # TODO add other orientations
    for word, orientation in words.items():
        # FIXME Save to specific dir, not just in cwd
        # Forward orientation
        if orientation == "forward":
            canvas = get_canvas_forward(word)
            # print(os.path.join(os.getcwd(), "words", f'{word}.png'))
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "backward":
            canvas = get_canvas_backward(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "down":
            canvas = get_canvas_down(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "up":
            canvas = get_canvas_up(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "backslash":
            canvas = get_canvas_backslash(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "rbackslash":
            canvas = get_canvas_rbackslash(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "forwardslash":
            canvas = get_canvas_forwardslash(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "rforwardslash":
            canvas = get_canvas_rforwardslash(word)
            cv2.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation is None:
            generate_all_canvases(word)


def update_image_with_new_match(reference_image, marking="rectangle", threshold=None, test_only=False):
    """
    Using a given reference image location, uses OpenCV2 to find the closest matching area on the puzzle
    and draw a rectangle around it
    """
    img_rgb = cv2.imread((os.path.join(os.getcwd(), "PuzzleSearchResults", "original.png")))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(reference_image, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    if threshold is None:
        # Function to narrow down the location of the desired word.
        threshold = 0.0
        loc = np.where(res >= threshold)
        results = len(list(zip(*loc[::-1])))
        max_precision_reached = False
        max_precision_attempts = 10
        precision_attempts = 0
        increments = (0.1, 0.05, 0.01, 0.001, 0.0001)
        direction = itertools.cycle(('up', 'down'))
        current_direction = next(direction)
        current_precision = 0
        previous_matches = -1
        while results != 1 and max_precision_attempts != precision_attempts:
            if current_direction == 'up':
                while results != 0:
                    threshold += increments[current_precision]

                    loc = np.where(res >= threshold)
                    results = len(list(zip(*loc[::-1])))
                    if results != 0:
                        previous_matches = results

                threshold -= increments[current_precision]

            elif current_direction == 'down':
                while results <= previous_matches:
                    threshold -= increments[current_precision]

                    loc = np.where(res >= threshold)
                    results = len(list(zip(*loc[::-1])))

                threshold += increments[current_precision]

            if current_precision >= len(increments) - 1:
                max_precision_reached = True

            if not max_precision_reached:
                current_precision += 1
            else:
                precision_attempts += 1

            current_direction = next(direction)
            loc = np.where(res >= threshold)
            results = len(list(zip(*loc[::-1])))
        if test_only:
            return threshold
    else:
        # If a threshold is given, no time should be wasted testing values
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

    # print(loc)
    # Color format: (b, g, r)
    color_choices = ((0, 0, 255),  # red
                     (0, 255, 0),  # green
                     (255, 0, 0),  # blue
                     (0, 128, 255),  # orange
                     (255, 128, 0),  # light blue
                     (255, 51, 153),  # red
                     (255, 51, 255),  # magenta
                     (102, 51, 0),  # brown
                     (0, 153, 0),  # dark green
                     (229, 229, 0),  # cyan
                     (119, 119, 119),  # mid-dark gray
                     (130, 130, 33),  # dark cyan
                     (100, 100, 239),  # salmon
                     (188, 135, 100),  # faded dark blue
                     (61, 165, 117),  # faded green
                     (56, 56, 56),  # dark gray
                     )

    # Once the locations have been narrowed down, a color may be chosen and rectangles drawn surrounding
    # the word on the puzzle
    img_rgb = cv2.imread((os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png")))
    selected_color = random.choice(color_choices)
    for pt in zip(*loc[::-1]):
        if marking == "rectangle":
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), selected_color, 2)
        elif marking == "line":
            cv2.line(img_rgb, pt, (pt[0] + w, pt[1] + h), selected_color, 2)
        elif marking == "fline":
            p = ((pt[0], pt[1] + h), (pt[0] + w, pt[1]))
            cv2.line(img_rgb, p[0], p[1], (0, 0, 255), 2)

    cv2.imwrite(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"), img_rgb)


def draw_results():
    """
    Search through .../words for all word images generated and match them on the puzzle by calling
    update_image_with_new_match
    """
    # Count how many files contain a word. If a match was not found, there will be more than one and should be handled
    # accordingly
    word_image_count = {}
    for root, _, files in os.walk(os.path.join(os.getcwd(), "words")):
        for name in files:
            separated_name = name.split("_")
            if separated_name[0] in word_image_count:
                word_image_count[separated_name[0]] += 1
            else:
                word_image_count[separated_name[0]] = 1

    processed_words = set()
    for root, _, files in os.walk(os.path.join(os.getcwd(), "words")):
        for name in files:
            path = os.path.join(root, name)
            print("Working on:", path)
            separated_name = name.split("_")
            if word_image_count[separated_name[0]] == 1:
                if "forwardslash" in separated_name[-1]:
                    update_image_with_new_match(path, "fline")
                elif "slash" in separated_name[-1]:
                    update_image_with_new_match(path, "line")
                else:
                    update_image_with_new_match(path, "rectangle")
            else:
                if separated_name[0] not in processed_words:
                    best_orientation, threshold = find_best_match(separated_name[0])
                    ending = str(separated_name[0]) + "_" + best_orientation + ".png"
                    update_image_with_new_match(os.path.join(root, ending), threshold=threshold)
                    processed_words.add(separated_name[0])


def find_best_match(word):
    paths = (os.path.join(os.getcwd(), "words", word, f"{word}_forward.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_backward.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_up.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_down.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_backslash.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_rbackslash.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_forwardslash.png"),
             os.path.join(os.getcwd(), "words", word, f"{word}_rforwardslash.png"))

    threshold_results = {"forward": None,
                         "backward": None,
                         "up": None,
                         "down": None,
                         "backslash": None,
                         "rbackslash": None,
                         "forwardslash": None,
                         "rforwardslash": None}

    search_order = ("forward", "backward", "up", "down", "backslash", "rbackslash", "forwardslash", "rforwardslash")
    for num, path in enumerate(paths):
        threshold_results[search_order[num]] = update_image_with_new_match(path, test_only=True)

    best_orientation = max(threshold_results, key=lambda key: threshold_results[key])
    return best_orientation, threshold_results[best_orientation]
