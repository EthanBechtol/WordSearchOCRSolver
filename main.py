import cv2
import pytesseract
import os
import argparse
from timeit import default_timer as timer

from file_processing import delete_files
from image_processing import draw_results, resize_image
from wordarray_processing import preprocess_puzzle


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

    img = cv2.imread(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))
    cv2.imwrite(os.path.join(os.getcwd(), "PuzzleSearchResults", "original.png"), img)
    preprocess_puzzle(words, os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))

    draw_results()
    if out is not None:
        img = cv2.imread(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))
        cv2.imwrite(out, img)

    if display:
        img = cv2.imread(os.path.join(os.getcwd(), "PuzzleSearchResults", "working.png"))
        cv2.namedWindow('Puzzle Search Results', cv2.WINDOW_NORMAL)
        cv2.imshow('Puzzle Search Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    delete_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", nargs=1, help="Specify the puzzle image path to solve.")
    parser.add_argument("-words", nargs='+', help="List the words you would like the program to find.")
    parser.add_argument("-out", nargs=1, help="Select a destination for the output.")
    parser.add_argument("-display", action='store_true', help="If added, will display any results in a new window.")
    parser.add_argument("-tesseract", nargs=1, help="Specify the location of your tesseract.exe file.")

    args, unknown_args = parser.parse_known_args()

    if not args.file or not args.words:
        print("ERROR: You must supply a file location and words to search for!")
        parser.print_help()
    else:
        if args.tesseract:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract[0]
        out = args.out[0] if args.out else None
        start = timer()
        try:
            run(args.file[0], set(args.words), args.display, out)
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract was not found! Please use -tesseract [dir] to specify a directory.")
            delete_files()
        end = timer()
        print("TIME ELAPSED:", end - start)
