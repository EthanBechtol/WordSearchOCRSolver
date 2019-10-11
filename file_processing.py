import os
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
        # print("Made path at " + os.getcwd() + f"/words/{word}")
    except Exception as e:
        print("ERROR: Could not make directory at:", os.path.join(os.getcwd(), "words", word), e)


def delete_files():
    """
    Remove the files used during this program's operation.
    Currently: .../PuzzleSearchResults and .../words.
    These have already been checked to be nonexistent prior to program start, so they are safe to delete.
    """
    shutil.rmtree(os.path.join(os.getcwd(), "PuzzleSearchResults"))
    shutil.rmtree(os.path.join(os.getcwd(), "words"))
