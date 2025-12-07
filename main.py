import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Tile import Tile
from Pixel import Pixel
#from animation_generator import simulate_solve_puzzle, generate_puzzle_animation
from solver import simulate_solve_puzzle
from animation_generator import generate_puzzle_animation


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
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        tiles = []
        numOfTiles = 0
        # loop through all edges for all shapes
        for i, contour in enumerate(contours):
            # approximate borders
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            sides = len(approx)

            if sides == 4:
                # Flatten contour points
                n = approx.ravel()
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Crop the tile image from the original image
                tile_img = img[y:y+h, x:x+w].copy()
                
                # Adjust coordinates relative to cropped tile
                coordinates = [(pt[0] - x, pt[1] - y) for pt in approx.reshape(-1, 2)]
                
                # Collect edge pixels relative to the tile
                pixels = []
                for p in contour:
                    px, py = p[0]
                    color = img[py, px]
                    pixels.append(Pixel(px - x, py - y, color[0], color[1], color[2]))
                
                # Draw borders and labels (optional)
                cv2.drawContours(img, [approx], 0, (255, 255, 255), 2)
                numOfTiles += 1
                cv2.putText(img, f"Tile {numOfTiles}", (n[0], n[1]+20), font, 0.4, (0, 255, 255))

                # Create Tile object
                tiles.append(Tile(corners=coordinates, edges=pixels, image=tile_img))


        print('Number of tiles found: ', numOfTiles)
        # cv2.imshow('image', img)
        # # press q to quite the picture and go through next one
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # =======Image Matching===========
        simulate_solve_puzzle(tiles, img)  # TODO: this function needs to be rewrite

        # =======Animation Generation=======
        output_name = os.path.splitext(image)[0] + "_solution.gif"
        output_path = "outputs/"+output_name
        os.makedirs("outputs", exist_ok=True)
        print(output_name)
        generate_puzzle_animation(tiles, img, output_filename=output_path)
