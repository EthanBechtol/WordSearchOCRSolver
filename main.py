import cv2 as cv
import pytesseract
import numpy as np
import os
from PIL import Image
import argparse
import itertools
import random
import shutil


first_dir_pass = True
safe_to_delete = False
def prepare_dir(word):
    """Create a directory if it doesn't already exist to store resources generated in this program"""
    global first_dir_pass
    global safe_to_delete

    try:
        os.mkdir(os.path.join(os.getcwd(), "words"))
    except FileExistsError:
        if first_dir_pass:
            print("CRITICAL ERROR:" + os.path.join(os.getcwd(), "words") +
                  "already exists prior to program start and cannot safely delete upon completion.")
            # raise FileExistsError
            exit(-1)
    else:
        # The directory did not exist prior to starting the program, so it is safe to delete it
        # once the program finishes.
        first_dir_pass = False
        safe_to_delete = True

    try:
        os.mkdir(os.path.join(os.getcwd(), "words", word))
        print("Made path at " + os.getcwd() + f"/words/{word}")
    except Exception as e:
        print("Something went wrong!", e)


def get_canvas_forward(word: str):  # left to right
    """Create & return an image containing the given word on a blank canvas to be used for matching"""
    font = cv.FONT_HERSHEY_DUPLEX

    word = str(word)
    im_width = len(word) * 72 - 28
    canvas = 255 * np.ones(shape=[50, im_width, 3], dtype=np.uint8)

    curr_x_pos = 0
    for letter in word.upper():
        if letter == 'I':
            cv.putText(canvas, letter, (curr_x_pos + 13, 45), font, 2, (0, 0, 0), 2, cv.LINE_4)

        else:
            cv.putText(canvas, letter, (curr_x_pos, 45), font, 2, (0, 0, 0), 2, cv.LINE_4)
        curr_x_pos += 72

    return canvas


def get_canvas_backward(word: str):  # right to left
    return get_canvas_forward(word[::-1])


def get_canvas_down(word: str):  # top to bottom
    letter_spacing = 36
    letter_height = 38
    padding = 18
    font = cv.FONT_HERSHEY_DUPLEX

    word = str(word.upper())
    im_height = len(word) * (letter_height + letter_spacing) - letter_spacing + padding * 2
    canvas = 255 * np.ones(shape=[im_height, 55, 3], dtype=np.uint8)

    current_position = padding + letter_height
    for letter in word:
        if letter == 'I':
            cv.putText(canvas, letter, (14, current_position), font, 2, (0, 0, 0), 2, cv.LINE_4)
        else:
            cv.putText(canvas, letter, (2, current_position), font, 2, (0, 0, 0), 2, cv.LINE_4)
        current_position += letter_height + letter_spacing

    return canvas


def get_canvas_up(word: str):  # bottom to top
    return get_canvas_down(word[::-1])


def get_canvas_backslash(word: str):  # top-left to bottom-right
    letter_height_spacing = 36
    letter_height = 38
    padding_height = 18

    letter_width = 42
    font = cv.FONT_HERSHEY_DUPLEX

    word = str(word.upper())
    im_height = len(word) * (letter_height + letter_height_spacing) - letter_height_spacing + padding_height * 2
    im_width = len(word) * 72 - 24
    canvas = 255 * np.zeros(shape=[im_height, im_width, 4], dtype=np.uint8)

    current_position_y = padding_height + letter_height
    current_position_x = 0
    for letter in word:
        if letter == '$':
            cv.putText(canvas, letter, (current_position_x, current_position_y), font, 2, (0, 0, 0, 255), 2, cv.LINE_4)
        else:
            rect_pt1 = (current_position_x - 13, current_position_y + 20)
            rect_pt2 = (current_position_x + letter_width + 13, current_position_y - letter_height - 20)
            cv.rectangle(canvas, rect_pt1, rect_pt2, (255, 255, 255, 255), -1)
            cv.putText(canvas, letter, (current_position_x, current_position_y), font, 2, (0, 0, 0, 255), 2, cv.LINE_4)
        current_position_y += letter_height + letter_height_spacing
        current_position_x += 72

    cv.imwrite("backslash_test.png", canvas)
    return canvas


def get_canvas_rbackslash(word: str):  # bottom-right to top-left
    return get_canvas_backslash(word[::-1])


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
            print(os.path.join(os.getcwd(), "words", f'{word}.png'))
            cv.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "backward":
            canvas = get_canvas_backward(word)
            cv.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "down":
            canvas = get_canvas_down(word)
            cv.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "up":
            canvas = get_canvas_up(word)
            cv.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "backslash":
            canvas = get_canvas_backslash(word)
            cv.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)

        elif orientation == "rbackslash":
            canvas = get_canvas_rbackslash(word)
            cv.imwrite(os.path.join(os.getcwd(), "words", word, f'{word}_{orientation}.png'), canvas)


def get_np_filled(input_data):
    """Returns a valid numpy array given an uneven 2D array by substituting blank spaces with zeros"""
    separated = []
    for line in input_data:
        separated.append(list(line))

    # Make the data into a uniform square with 0's as filler
    data = np.array([np.array(line) for line in separated])
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:, None]
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def get_columns(puzzle: list):
    """Return a list of lists containing the columns of the given puzzle"""
    arr = get_np_filled(puzzle)
    columns = []
    for column in range(arr.shape[1]):
        columns.append(arr[:, column].tolist())

    return columns


def search_for_word(puzzle: list, word: str, diags_backslash, diags_forwardslash):
    """
    Searches the puzzle for the given words and creates a dictionary containing the word and the orientation it was
    found in
    """
    # Search for word in horizontal combinations
    target = word.upper()
    for line in puzzle:
        if target in line:
            return "forward"
        elif target in line[::-1]:
            return "backward"

    # Search for word in vertical combinations
    columns_separated = get_columns(puzzle)
    # TODO if last column, 0s could be in the middle of the word
    for column in columns_separated:
        while 0 in column:
            column.remove(0)

    columns = ["".join(column) for column in columns_separated]
    for column in columns:
        if target in column:
            return "down"  # top to bottom
        elif target in column[::-1]:
            return "up"  # bottom to top

    # Search for word in diagonal combinations
    for line in diags_backslash:
        if target in line:
            return "backslash"  # top-left to bot-right
        elif target in line[::-1]:
            return "rbackslash"  # bot-right to top-left

    for line in diags_forwardslash:
        if target in line:
            return "forwardslash"  # bot-left to top-right
        elif target in line[::-1]:
            return "rforwardslash"  # top-right to bot-left

    return None


def get_diagonals(puzzle: list):
    """Return 2 lists containing every diagonal combination in the given puzzle for use in search functions"""
    arr = get_np_filled(puzzle)
    diags_forwardslash = [arr[::-1, :].diagonal(i).tolist() for i in range(-arr.shape[0]+1, arr.shape[1])]
    diags_backslash = [arr.diagonal(i).tolist() for i in range(arr.shape[1]-1, -arr.shape[0], -1)]

    print(arr)
    # print(diags_backslash)
    # print(diags_forwardslash)

    return diags_backslash, diags_forwardslash


def preprocess_puzzle(words: set, puzzle_location: str):
    """
    Uses Google's Tesseract engine to first try and get a text representation of the puzzle and find the orientations
    of the given words in order to create reference images with which matches may be made on the puzzle image
    """
    # Create directories for the words to be used
    for word in words:
        prepare_dir(word)

    image = Image.open(puzzle_location)
    raw_list = pytesseract.image_to_string(image, config='--psm 6').splitlines()
    fixed_list = []
    for line in raw_list:
        fixed_list.append(line.rstrip(' |.-')
                          .replace(' ', '')
                          .replace('|', 'I')
                          .replace('1', 'I')
                          .replace('!', 'I')
                          .replace(')', ''))  # FIXME Maybe not this one?

    diags_backslash, diags_forwardslash = get_diagonals(fixed_list)

    # Remove any filler 0s from the sequence to join them
    for diag in diags_backslash:
        while 0 in diag:
            diag.remove(0)

    for diag in diags_forwardslash:
        while 0 in diag:
            diag.remove(0)

    diagstrings_backslash = ["".join(diag) for diag in diags_backslash]
    diagstrings_forwardslash = ["".join(diag) for diag in diags_forwardslash]
    # print(diagstrings_forwardslash)

    # Search the puzzle for target words and return their found orientation (None if not found)
    word_orientations = {}
    for word in words:
        found_orientation = search_for_word(fixed_list, word, diagstrings_backslash, diagstrings_forwardslash)
        word_orientations[word] = found_orientation

    print(word_orientations)
    prepare_word_images(word_orientations)


def update_image_with_new_match(reference_image, marking="rectangle"):
    """
    Using a given reference image location, uses OpenCV2 to find the closest matching area on the puzzle
    and draw a rectangle around it
    """
    img_rgb = cv.imread((os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png")))
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(reference_image, 0)
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

    # Function to narrow down the location of the desired word.
    threshold = 0.0
    loc = np.where(res >= threshold)
    results = len(list(zip(*loc[::-1])))
    max_precision_reached = False
    increments = (0.1, 0.05, 0.01, 0.001)
    direction = itertools.cycle(('up', 'down'))
    current_direction = next(direction)
    current_precision = 0
    previous_matches = -1
    while results != 1 and not max_precision_reached:
        # TODO check function operation
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

        current_precision += 1
        if current_precision >= len(increments):
            max_precision_reached = True

        current_direction = next(direction)
        loc = np.where(res >= threshold)
        results = len(list(zip(*loc[::-1])))

    print(loc)
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
                     )

    # Once the locations have been narrowed down, a color may be chosen and rectangles drawn surrounding
    # the word on the puzzle
    selected_color = random.choice(color_choices)
    for pt in zip(*loc[::-1]):
        if marking == "rectangle":
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), selected_color, 2)
        elif marking == "line":
            cv.line(img_rgb, pt, (pt[0] + w, pt[1] + h), selected_color, 2)

    cv.imwrite(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"), img_rgb)


def draw_results():
    """
    Search through .../words for all word images generated and match them on the puzzle by calling
    update_image_with_new_match
    """
    for root, _, files in os.walk(os.path.join(os.getcwd(), "words")):
        for name in files:
            path = os.path.join(root, name)
            print("Working on:", path)
            separated_name = name.split("_")
            print(separated_name)
            if "slash" in separated_name[-1]:
                update_image_with_new_match(path, "line")
            else:
                update_image_with_new_match(path, "rectangle")


def delete_files():
    """
    Remove the files used during this program's operation.
    Currently: .../PuzzleSearchResults and .../words.
    These have already been checked to be nonexistent prior to program start, so they are safe to delete.
    """
    shutil.rmtree(os.path.join(os.getcwd(), "PuzzleSearchResults"))
    shutil.rmtree(os.path.join(os.getcwd(), "words"))


def resize_image(inFile: str):
    src = cv.imread(inFile)
    dst = os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png")
    dsize = (1_286, 1_116)
    new_img = cv.resize(src, dsize)
    cv.imwrite(dst, new_img)


def run(inFile: str, words: set, display: bool = False, out: str = None):
    try:
        os.mkdir(os.path.join(os.getcwd(), "PuzzleSearchResults"))
    except FileExistsError:
        print("CRITICAL ERROR:" + os.path.join(os.getcwd(), "PuzzleSearchResults"),
              "already exists prior to program start. Unsure if it is safe to write to.")
        # raise FileExistsError
        exit(-1)
    else:
        resize_image(inFile)

    preprocess_puzzle(words, os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))

    draw_results()
    if out is not None:
        img = cv.imread(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))
        cv.imwrite(out, img)

    if display:
        # FIXME Odd issue sometimes causing a single side of the rectangles to not be displayed.
        #  Increasing rectangle thickness from 1 to 2 fixes its display,
        #  yet issue is never present when opening the saved file.
        img = cv.imread(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))
        cv.namedWindow('Puzzle Search Results', cv.WINDOW_NORMAL)
        cv.imshow('Puzzle Search Results', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    delete_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", nargs=1, help="Specify the puzzle image path to solve.")
    parser.add_argument("-words", nargs='+', help="List the words you would like the program to find.")
    parser.add_argument("-out", nargs=1, help="Select a destination for the output.")
    parser.add_argument("-display", action='store_true', help="If added, will display any results in a new window.")

    args, unknown_args = parser.parse_known_args()

    if not args.file or not args.words:
        print("ERROR: You must supply a file location and words to search for!")
        parser.print_help()
    else:
        out = args.out[0] if args.out else None
        run(args.file[0], set(args.words), args.display, out)
