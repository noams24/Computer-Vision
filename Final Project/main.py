import tensorflow as tf
from tensorflow.image import random_contrast, random_brightness, random_saturation, random_hue
from tensorflow.keras.models import model_from_json
import json
import h5py
import numpy as np
import cv2
from PIL import Image
import pandas as pd


class RandomColorDistortion(tf.keras.layers.Layer):
    def __init__(self, contrast_range=(1, 6), brighness_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=0.5,
                 **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.contrast_range = contrast_range
        self.brighness_range = brighness_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, images, training=None):
        if training:
            images = random_contrast(images, self.contrast_range[0], self.contrast_range[1])
            images = random_brightness(images, self.brighness_range[0], self.brighness_range[1])
            images = random_saturation(images, self.saturation_range[0], self.saturation_range[1])
            images = random_hue(images, self.hue_delta)
            images = tf.clip_by_value(images, 0, 1)
        return images


def concentrateImg(src_img, src_BB, target_BB_res=(44, 44), hmargin=10, vmargin=10):
    # target boundry box (for the homography)
    target_BB = np.array([[hmargin, vmargin],
                          [hmargin + target_BB_res[0], vmargin],
                          [hmargin + target_BB_res[0], vmargin + target_BB_res[1]],
                          [hmargin, vmargin + target_BB_res[1]]]).transpose()
    # resolution of the target concentrated image (including margins)
    target_img_res = (2 * hmargin + target_BB_res[0],
                      2 * vmargin + target_BB_res[1])
    # calculate the homography from original BB to target BB
    H = cv2.findHomography(src_BB.transpose(), target_BB.transpose())[0]
    H = cv2.warpPerspective(src_img, H, target_img_res)
    img = Image.fromarray(H)
    return img


def find_name(num):
    if num == 0:
        return 'Alex Brush'
    if num == 1:
        return 'Open Sans'
    if num == 2:
        return 'Sansation'
    if num == 3:
        return 'Titillium Web'
    return 'Ubuntu Mono'


if __name__ == "__main__":
    # loading the dataset:
    db = h5py.File('SynthText_test.h5')
    im_names = list(db['data'].keys())

    df = {'image': [], 'char': [], "Open Sans": [], "Sansation": [], "Titillium Web": [],
          "Ubuntu Mono": [], "Alex Brush": []}

    # loading the probability letter dictionary file:
    with open('model/letters_probabilty.json', 'r') as openfile:
        letter_probabilties = json.load(openfile)

    # loading the model:

    with open('model/model.json', 'r') as json_file:
        json_model = json_file.read()
    model = model_from_json(json_model, custom_objects={'RandomColorDistortion': RandomColorDistortion})
    model.load_weights('model/model.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    for im in im_names[:]:
        i = 0
        img = db['data'][im][:]
        txt = db['data'][im].attrs['txt']
        charBB = db['data'][im].attrs['charBB']
        for word in txt:
            result = np.array([0, 0, 0, 0, 0])
            for letter in word:
                # creating new row in the data frame:
                df['image'].append(str(im))
                df['char'].append(chr(letter))
                # preproccess the image
                try:
                    proccessed_img = np.array(concentrateImg(img, charBB[:, :, i]))
                    proccessed_img = proccessed_img / 255
                    proccessed_img = proccessed_img.reshape(-1, 64, 64, 3)
                except:
                    i += 1
                    continue
                # get the result of the model
                if letter_probabilties.get(chr(letter)) is not None:
                    result += model(proccessed_img) * letter_probabilties[chr(letter)] / (
                            2 - letter_probabilties[chr(letter)])
                i += 1
            predicted_font = find_name(np.argmax(result))

            # add the result to the data frame
            for j in range(len(word)):
                df['Alex Brush'].append(1 if predicted_font == 'Alex Brush' else 0)
                df['Open Sans'].append(1 if predicted_font == 'Open Sans' else 0)
                df['Sansation'].append(1 if predicted_font == 'Sansation' else 0)
                df['Titillium Web'].append(1 if predicted_font == 'Titillium Web' else 0)
                df['Ubuntu Mono'].append(1 if predicted_font == 'Ubuntu Mono' else 0)

    df = pd.DataFrame(df)
    df.to_csv('font_predictions.csv')
