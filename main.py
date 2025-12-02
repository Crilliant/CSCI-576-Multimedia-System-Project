import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Tile import Tile
from Pixel import Pixel
from animation_generator import simulate_solve_puzzle, generate_puzzle_animation

font = cv2.FONT_HERSHEY_COMPLEX

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    images = ["starry_night_translate.png", "starry_night_rotate.png", "starry_night_translate.rgb", "starry_night_rotate.rgb",
              "mona_lisa_translate.png", "mona_lisa_rotate.png", "mona_lisa_translate.rgb", "mona_lisa_rotate.rgb",
              "sample1_translate.png", "sample1_rotate.png", "sample1_translate.rgb", "sample1_rotate.rgb",
              "sample2_translate.png", "sample2_rotate.png", "sample2_translate.rgb", "sample2_rotate.rgb",
              "sample3_translate.png", "sample3_rotate.png", "sample3_translate.rgb", "sample3_rotate.rgb"]
    # go through all images in the samples folder
    for image in images:
        print("Image: ", image)
        imagePath = "samples/" + image
        splitup = os.path.splitext(image)
        # get file extension
        fileExtension = splitup[1]
        # if .rgb file convert to a readable array
        if fileExtension == ".rgb":
            raw = np.fromfile(imagePath, dtype=np.uint8)
            if raw is None:
                 raise ValueError("Unable to read rgb file: " + imagePath)
            #convert rgb to bgr
            reshaped = raw.reshape(3, 800, 800,)
            transposed = np.transpose(reshaped, (1,2,0))
            img = cv2.cvtColor(transposed, cv2.COLOR_RGB2BGR)
        # else image must be .png so open it
        else:
            img = cv2.imread(imagePath)
            if img is None:
                raise ValueError("Unable to read png file: " + imagePath)

        # convert to grayscale to better find edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # declare lower threashold as black and upper as white
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # find all shapes & edges
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        tiles = []
        numOfTiles = 0
        # loop through all edges for all shapes
        for i, contour in enumerate(contours):
            # approximate borders
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            sides = len(approx)
            # if it iss a square or rectangle draw the borders and print corner coordinates
            if sides == 4:
                cv2.drawContours(img, [contour], 0, (255, 255, 255), 1)
                numOfTiles += 1
                # make into 1d array
                n = approx.ravel()
                # print(n)
                coordinates = []
                i = 0
                for j in n:
                    if i % 2 == 0:
                        # represent corner coordinates
                        x, y = n[i], n[i + 1]
                        coordinates.append((x,y))
                        coord = f"({x}, {y})"
                        cv2.putText(img, coord, (x, y), font, 0.4, (0, 255, 0))
                        # print("tile ", numOfTiles, " corner coordinate: ", coord)
                    i += 1
                # for all pixels in the edge print the coordinates
                pixels = []
                for p in contour:
                    x, y = p[0]
                    color = img[y,x]
                    pixel = Pixel(x, y, color[0], color[1], color[2])
                    pixels.append(pixel)
                    pix = f" edge pixels coordinates: ({x}, {y})"
                    # print("tile ", numOfTiles, pix)
                tiles.append(Tile(coordinates, pixels)) # list of all tiles with their coordinates and edges

        print('Number of tiles found: ', numOfTiles)
        # cv2.imshow('image', img)
        # # press q to quite the picture and go through next one
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Animation part
        simulate_solve_puzzle(tiles, img)  # TODO: this function needs to be rewrite

        output_name = os.path.splitext(image)[0] + "_solution.gif"
        output_path = "outputs/"+output_name
        os.makedirs("outputs", exist_ok=True)
        print(output_name)
        generate_puzzle_animation(tiles, img, output_filename=output_path)

