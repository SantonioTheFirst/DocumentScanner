import streamlit as st
import numpy as np
import cv2
import time


st.set_page_config(page_title='Document scanner demo', page_icon=':document:')


width: int = 512
height: int = 512
file = None


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    img: np.ndarray = cv2.resize(img, (width, height))
    return img
    
    
def to_grayscale(img: np.ndarray) -> np.ndarray:
    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
    
def add_gaussian_blur(img: np.ndarray) -> np.ndarray:
    img: np.ndarray = cv2.GaussianBlur(img, (5, 5), 1)
    return img
    
    
def apply_Canny_filter(img: np.ndarray, threshold1: int = 50, threshold2: int = 100, aperture_size: int = 3) -> np.ndarray:
    img: np.ndarray = cv2.Canny(
        image=img,
        threshold1=threshold1,
        threshold2=threshold2,
        apertureSize=3,
        L2gradient=True
    )
    return img
    
    
def dilate_and_erode(img: np.ndarray, kernel: np.ndarray = np.ones((3, 3)), d_iter: int = 1, e_iter: int = 1) -> np.ndarray:
    imgDial: np.ndarray = cv2.dilate(img, kernel, iterations=d_iter)
    imgThreshold: np.ndarray = cv2.erode(imgDial, kernel, iterations=e_iter)
    return imgThreshold
    
    
def get_all_contours(img: np.ndarray, imgThreshold: np.ndarray) -> tuple[tuple[np.ndarray], np.ndarray]:
    imgContours: np.ndarray = img.copy()
    contours: tuple[np.ndarray, None] = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: np.ndarray = contours[0]
    allContours: np.ndarray = cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    return contours, allContours
    
    
def get_top_contours(img: np.ndarray, contours: np.ndarray, contour_area_threshold: float = 10000) -> tuple[dict[str, np.ndarray], np.ndarray]:
    top_contours: dict[str, np.ndarray] = {}
    imgContours: np.ndarray = img.copy()
    # print(contours.shape)
    for contour in contours:
        # print(type(contour), contour.shape)
        area: float = cv2.contourArea(contour)
        if area > contour_area_threshold:
            peri = cv2.arcLength(contour, True)
            approx: np.ndarray = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # print(len(approx))
            if len(approx) in set([4, 5]):
                top_contours[f'{area}_{len(approx)}'] = approx
    imgContours: np.ndarray = cv2.drawContours(imgContours, list(top_contours.values()), -1, (0, 255, 0), 20)
    return top_contours, imgContours
    
    
def reorder(myPoints: list) -> np.ndarray:
    result: list[np.ndarray] = []
    for points in myPoints:
        points: np.ndarray = points.reshape((4, 2))
        pointsNew: np.ndarray = np.zeros((4, 1, 2), dtype=np.int32)
        add: np.ndarray = points.sum(1)

        pointsNew[0] = points[np.argmin(add)]
        pointsNew[3] = points[np.argmax(add)]
        diff: np.ndarray = np.diff(points, axis=1)
        pointsNew[1] = points[np.argmin(diff)]
        pointsNew[2] = points[np.argmax(diff)]
        result.append(pointsNew)
    return np.array(result, dtype=np.float32)
    
    
def transform(img: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    matrix: np.ndarray = cv2.getPerspectiveTransform(pts1, pts2)
    result: np.ndarray = cv2.warpPerspective(img, matrix, (width, height))
    return result
    
    
def get_clipped_img(img: np.ndarray, width: int, height: int, threshold1: int = 50, threshold2: int = 100, verbose: bool = True) -> np.ndarray:
    #resizedImg: np.ndarray = resize(img, width, height)
    grayImg: np.ndarray = to_grayscale(img)
    blurredImg: np.ndarray = add_gaussian_blur(grayImg)
    CannyImg: np.ndarray = apply_Canny_filter(blurredImg, threshold1, threshold2)
    deImg: np.ndarray = dilate_and_erode(CannyImg)
    allContours: tuple[tuple[np.ndarray], np.ndarray]  = get_all_contours(img, deImg)
    contoursImg: np.ndarray = allContours[1]
    allContours: tuple[np.ndarray] = allContours[0]
    topContours: tuple[dict[str, np.ndarray], np.ndarray] = get_top_contours(img, allContours)
    imgContours: np.ndarray = topContours[1]
    topContours: dict[str, np.ndarray] = topContours[0]
    try:
        print(reorder(list(topContours.values())))
        pts1: np.ndarray = reorder(list(topContours.values()))[0]
        pts2: np.ndarray = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) 
        transformedImg: np.ndarray = transform(img, pts1, pts2)
        if verbose:
            images = [
           # resizedImg,
                grayImg, 
                blurredImg, 
                CannyImg, 
                deImg,
                contoursImg, 
                imgContours, 
                transformedImg
            ]
            captions = [
           # 'Original resized image', 
                'Gray image', 
                'Blurred image', 
                'Canny filter', 
                'Dilated and eroded image', 
                'All contours', 
                'Top contours', 
                'Clipped and transformed image'
            ]
            st.image(images, caption=captions)
        return transformedImg
    except:
        st.info('Cannot process this image, change your thresholds.') 

    
'''
# Hello!

This toy website allows you to interact with image document scanner service.

Just upload your image with document below!
'''

file = st.file_uploader('Upload your documents', accept_multiple_files=False)
if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    threshold1: int = st.slider('Threshold 1', 0, 255, 50, disabled=False)
    threshold2: int = st.slider('Threshold 2', 0, 255, 100, disabled=False)
    get_clipped_img(img, width, height, threshold1, threshold2)
