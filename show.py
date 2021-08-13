import tkinter as tk
import os
import random
import cv2
from PIL import Image, ImageTk

image_path = 'data\\Robbins\\images'
label_path_class = 'data\\Robbins\\classifier\\labels'
label_path = 'data\\Robbins\\labels'
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
line_width = 2
shuffle = True
descending = True

with open('data\Robbins\classifier\image_list') as f:
    lines = f.readlines()
    img_list = [line.rstrip('\n') for line in lines]

if not descending:
    img_list.reverse()

slash = os.sep
img_format = img_list[0].split('.')[-1]
if shuffle:
    random.shuffle(img_list)
img_idx = -1
img_len = len(img_list)

window = tk.Tk()
tk_label = tk.Label(window)
tk_label.pack(padx=10, pady=10) 

def xywh2xyxy(x, y, w, h, dtype=float, scale=(1, 1)):
    x, y, w, h = float(x) * scale[0], float(y) * scale[1], float(w) * scale[0], float(h) * scale[1]
    return dtype(x - w / 2), dtype(y - h / 2), dtype(x + w / 2), dtype(y + h / 2)

def get_label_list(label_file):
    # Classified
    try:
        f = open(label_path_class + slash + label_file, 'r')
        label_list = [line.split(' ') for line in f.readlines()]
        ids = [label[-1].rstrip('\n') for label in label_list]
        f.close()
    except:
        label_list = []

    # Normal
    f = open(label_path + slash + label_file, 'r')
    for line in f.read().splitlines():
        line = line.split(' ')
        if line[-1].rstrip('\n') not in ids:
            label_list.append(line)
    f.close()

    return label_list

def show_next():
    global img_idx
    global img_list
    global tk_label
    global tk_img

    img_idx += 1
    if img_idx == img_len:
        exit()

    tk_label.destroy()


    img_name = img_list[img_idx]
    lbl_name = img_name.replace(img_format, 'txt')

    window.title('%d/%d, %s' % (img_idx + 1, img_len, img_name))

    img = cv2.imread(image_path + slash + img_name)

    img_shape = img.shape

    label_list = get_label_list(lbl_name)
    
    for label in label_list:
        x1, y1, x2, y2 = xywh2xyxy(*label[1:5], dtype=round, scale=img_shape[:2])

        if label[0] == '0':
            cv2.rectangle(img, (x1, y1), (x2, y2), red, line_width)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), green, line_width)
            cv2.putText(img, label[0], (x1, y1-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1, cv2.LINE_AA)
    
    pillow_img = Image.fromarray(img[:,:,::-1])
    tk_img = ImageTk.PhotoImage(image=pillow_img)
    tk_label = tk.Label(window, image=tk_img)
    tk_label.pack(padx=10, pady=10) 

show_next()

next_set = set(['bracketright', 'Down', 'Right'])
last_set = set(['bracketleft', 'Left', 'Up'])

def keyboard_action(event):
    key = event.keysym
    if key in last_set:
        global img_idx
        if img_idx > 0:
            img_idx -= 2
            show_next()
    elif key in next_set:
        show_next()
window.bind('<Key>', keyboard_action)

window.mainloop()