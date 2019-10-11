# WordSearchOCRSolver
A word search puzzle solver using Google's Tesseract-OCR engine as well as OpenCV 2 in order to solve a given puzzle screenshot.

Currently optimized for use on: \
http://word-search-puzzles.appspot.com/

NOTE: [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) must be installed for this program to work.

---
Usage
---
1. Go to http://word-search-puzzles.appspot.com/
2. Take a screenshot of the entire puzzle up to where the white border meets the gray background.
3. Use the following command line arguments to start the program with the screenshot location, and a list of words you would like the program to find.

usage: main.py [-file FILE] [-words WORDS [WORDS ...]] [-out OUT]
               [-display] [-tesseract TESSERACT]

####optional arguments:

| Argument   | Parameters       | Description
-------------|:----------------:|------------
| -file      | file             | Specify the path to the puzzle image.
| -words     | words [words...] | List one or more words you would like the program to find in the puzzle.
| -out       | out              | Specify a location for the program to save the final markedup puzzle to.
| -display   |                  | Opens a new window containing the marked search results.
| -tesseract | tesseract        | If the program throws a "TesseractNotFoundError", use this argument to specify the location of your tesseract.exe executable.

Sample usage:
-file TestPuzzle5.png -display -out outfilelocation.png -words ANAESTHESIA BRAVOS EDIFIES EXPURGATING FASHIONED GRANDSTANDED HAFNIUM IMBROGLIOS JOISTS LIAISED MINDFUL MOSS OVERRUNNING PADRE QUAYS SLEEPYHEAD STYE SUPEREGO TALLER VOTES YOLK YUKS ZIGZAGS ZOOS

Sample Input/Output
---

![SampleInput5](/TestPuzzle5.png)
<img src="https://github.com/EthanBechtol/WordSearchOCRSolverImages/blob/master/SampleOutput5.png" alt="Sample Output 5" width="641" height="561">
