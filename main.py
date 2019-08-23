import cv2 as cv
import pytesseract
import numpy as np
import os
from PIL import Image
import argparse


def prepare_dir(word):
    if not os.path.exists(os.getcwd() + "/words)"):
        try:
            os.mkdir(os.getcwd() + "/words")
        except Exception as e:
            print("Something went wrong!", e)

    if not os.path.exists(os.getcwd() + f"/words/{word}"):
        try:
            os.mkdir(os.getcwd() + f"/words/{word}")
            print("Made path at " + os.getcwd() + f"/words/{word}")
        except Exception as e:
            print("Something went wrong!", e)
    else:
        print('Exists')


def get_canvas_forward(word: str):
    char_width = 45
    char_spacing = 32
    font = cv.FONT_HERSHEY_DUPLEX

    word = str(word)
    im_width = len(word) * (char_width + char_spacing) - char_spacing
    canvas = 255 * np.ones(shape=[50, im_width, 3], dtype=np.uint8)

    formatted_word = ''
    for letter in word:
        formatted_word += letter.upper() + ' '
    formatted_word = formatted_word.rstrip()

    cv.putText(canvas, formatted_word, (0, 45), font, 2, (0, 0, 0), 2, cv.LINE_4)
    return canvas


def get_canvas_backward(word: str):
    return get_canvas_forward(word[::-1])


def prepare_word_images(words: dict):
    # TODO account for different spacing for larger letters (A, H, G, etc) and smaller letters (I, J, etc)
    for word, orientation in words.items():
        # FIXME Save to specific dir, not just in cwd
        # Forward orientation
        if orientation == "forward":
            canvas = get_canvas_forward(word)
            cv.imwrite('canvas.png', canvas)

        elif orientation == "backward":
            canvas = get_canvas_backward(word)
            cv.imwrite('canvas.png', canvas)


def get_np_filled(input_data):
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
    arr = get_np_filled(puzzle)
    columns = []
    for column in range(arr.shape[1]):
        columns.append(arr[:, column].tolist())

    return columns


def search_for_word(puzzle: list, word: str, diags_backslash, diags_forwardslash):
    # Horizontally
    target = word.upper()
    for line in puzzle:
        if target in line:
            return "forward"
        elif target in line[::-1]:
            return "backward"

    # Vertically
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

    # Diagonals
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
    arr = get_np_filled(puzzle)
    diags_forwardslash = [arr[::-1, :].diagonal(i).tolist() for i in range(-arr.shape[0]+1, arr.shape[1])]
    diags_backslash = [arr.diagonal(i).tolist() for i in range(arr.shape[1]-1, -arr.shape[0], -1)]

    # print(arr)
    # print(diags_backslash)
    # print(diags_forwardslash)

    return diags_backslash, diags_forwardslash


def preprocess_puzzle(words: set):
    # TODO Change how to select image.
    image = Image.open('TestPuzzle.png')
    raw_list = pytesseract.image_to_string(image, config='--psm 6').splitlines()
    fixed_list = []
    for line in raw_list:
        fixed_list.append(line.rstrip(' |.-')
                          .replace(' ', '')
                          .replace('|', 'I')
                          .replace('1', 'I')
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

    word_orientations = {}
    for word in words:
        found_orientation = search_for_word(fixed_list, word, diagstrings_backslash, diagstrings_forwardslash)
        word_orientations[word] = found_orientation

    # print(word_orientations)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs=1, help="Specify the puzzle image path to solve.")
    parser.add_argument("-out", nargs=1, help="Select a destination for the output.")
    parser.add_argument("-display", action='store_true')

