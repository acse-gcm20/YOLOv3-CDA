from PIL import Image
import numpy as np

def tiling(folder_name, file_name):
    imgPath = folder_name+file_name
    img = Image.open(imgPath)

    tileSize = 416

    # Image Numpy array
    imgArray = np.asarray(img)
    img_height, img_width = imgArray.shape
    remainder_x = img_width % tileSize
    remainder_y = img_height % tileSize

    # Generate padded array
    fullArray = np.zeros((img_height+(tileSize-remainder_y),
                        img_width+(tileSize-remainder_x)), dtype='uint8')
    fullArray[:img_height, :img_width] = imgArray
    height, width = fullArray.shape

    # Tiling dimensions
    num_tiles_x = int(np.floor(width / tileSize))
    num_tiles_y = int(np.floor(height / tileSize))
    num_tiles = num_tiles_x * num_tiles_y

    print('Original Image Size: {} x {}'.format(img_width, img_height),
        '\nPadded Image Size: {} x {}'.format(width, height),
        '\nImage contains {} tiles ({} x {})'.format(num_tiles, num_tiles_x, num_tiles_y))

    # Perform tiling, tiles saved as JPEGs
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):

            x_start = j * tileSize
            x_end = x_start + tileSize
            y_start = i * tileSize
            y_end = y_start + tileSize
            
            tileArray = fullArray[y_start:y_end, x_start:x_end]
            img = Image.fromarray(tileArray, mode='L')
            img.save('data/HiRISE/tiles/'+filename+'_{}_{}.jpg'.format(i, j))
