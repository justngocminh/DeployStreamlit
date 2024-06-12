import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_PATH = 'picture.jpg'
FONT_PATH = 'arial.ttf'

reader = easyocr.Reader(['en', 'vi'],  gpu=False)
result = reader.readtext(IMAGE_PATH)
print(result)
