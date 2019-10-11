import numpy as np
import pytesseract
from PIL import Image

from file_processing import prepare_dir
from image_processing import prepare_word_images


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
    diagonals_forwardslash = [arr[::-1, :].diagonal(i).tolist() for i in range(-arr.shape[0] + 1, arr.shape[1])]
    diagonals_backslash = [arr.diagonal(i).tolist() for i in range(arr.shape[1] - 1, -arr.shape[0], -1)]

    # print(arr)
    # print(diagonals_backslash)
    # print(diagonals_forwardslash)

    return diagonals_backslash, diagonals_forwardslash


def preprocess_puzzle(words: set, puzzle_location: str):
    """
    Uses Google's Tesseract engine to first try and get a text representation of the puzzle and find the orientations
    of the given words in order to create reference images with which matches may be made on the puzzle image
    """
    # Create directories for the words to be used
    for word in words:
        prepare_dir(word)

    image = Image.open(puzzle_location)

    raw_list = pytesseract.image_to_string(image,
                                           config='--psm 6 -c load_punc_dawg=F -c load_number_dawg=F -c load_unambig_dawg=F -c load_bigram_dawg=F -c load_fixed_length_dawgs=F -c load_freq_dawg=F -c load_system_dawg=F -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM').splitlines()
    fixed_list = []
    for line in raw_list:
        if line != '':
            # print(line)
            fixed_list.append(line.rstrip(' |.-'))

    diagonals_backslash, diagonals_forwardslash = get_diagonals(fixed_list)

    # Remove any filler 0s from the sequence to join them
    for diag in diagonals_backslash:
        while 0 in diag:
            diag.remove(0)

    for diag in diagonals_forwardslash:
        while 0 in diag:
            diag.remove(0)

    diagonal_strings_backslash = ["".join(diag) for diag in diagonals_backslash]
    diagonal_strings_forwardslash = ["".join(diag) for diag in diagonals_forwardslash]

    # Search the puzzle for target words and return their found orientation (None if not found)
    word_orientations = {}
    for word in words:
        found_orientation = search_for_word(fixed_list, word, diagonal_strings_backslash, diagonal_strings_forwardslash)
        word_orientations[word] = found_orientation

    # print(word_orientations)
    prepare_word_images(word_orientations)
