import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO('modelos/YoloHeavyV8.pt')
vinomodel = model.export(format ='openvino')