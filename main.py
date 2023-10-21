import streamlit as st
import numpy as np
import cv2
import time


st.set_page_config(page_title='Document scanner demo', page_icon=':document:')


width: int = 512
height: int = 512
file = None

def add_border(img: np.ndarray, border_size: int = 50, value: int = 255) -> np.ndarray:
    border = [border_size for i in range(4)]
    value = [value for i in range(3)]
    return cv2.copyMakeBorder(img, *border, cv2.BORDER_CONSTANT, value=value)


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    img: np.ndarray = cv2.resize(img, (width, height))
    return img
    
    
def to_grayscale(img: np.ndarray) -> np.ndarray:
    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
    
def add_gaussian_blur(img: np.ndarray) -> np.ndarray:
    img: np.ndarray = cv2.GaussianBlur(img, (5, 5), 1)
    return img
    
    
def apply_Canny_filter(img: np.ndarray, threshold1: int = 50, threshold2: int = 255, aperture_size: int = 3) -> np.ndarray:
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
    
    
def get_top_contours(img: np.ndarray, contours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top_contours: list[np.ndarray] = []
    imgContours: np.ndarray = img.copy()
    contour_area_threshold: float = (img.shape[0] * img.shape[1]) / 25.0
    st.info(contour_area_threshold)
    # print(contours.shape)
    for contour in contours:
        # print(type(contour), contour.shape)
        area: float = cv2.contourArea(contour)
        if area > contour_area_threshold:
            peri = cv2.arcLength(contour, True)
            approx: np.ndarray = cv2.approxPolyDP(contour, 0.01 * peri, True)
            # print(len(approx))
            if len(approx) in set([4, 5, 6, 7]):
                top_contours.append(approx)
    top_contours = np.array(top_contours)
    imgContours: np.ndarray = cv2.drawContours(imgContours, top_contours, -1, (0, 255, 0), 10)
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
    img border = add_border(img)
    #resizedImg: np.ndarray = resize(img, width, height)
    grayImg: np.ndarray = to_grayscale(imgBorder)
    blurredImg: np.ndarray = add_gaussian_blur(grayImg)
    CannyImg: np.ndarray = apply_Canny_filter(blurredImg, threshold1, threshold2)
    deImg: np.ndarray = dilate_and_erode(CannyImg)
    allContours: tuple[tuple[np.ndarray], np.ndarray]  = get_all_contours(img, deImg)
    contoursImg: np.ndarray = allContours[1]
    allContours: tuple[np.ndarray] = allContours[0]
    topContours: tuple[np.ndarray, np.ndarray] = get_top_contours(img, allContours)
    imgContours: np.ndarray = topContours[1]
    topContours: np.ndarray = topContours[0]
    #try:
       # st.warning(reorder(list(topContours.values())))
    pts1: np.ndarray = reorder(topContours)
    pts2: np.ndarray = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) 
    transformedImages: list[np.ndarray] = [transform(img, p, pts2) for p in pts1]
    if verbose:
        images = [
            imgBorder,
            grayImg, 
            blurredImg, 
            CannyImg, 
            deImg,
            contoursImg, 
            imgContours
        ]
        captions = [
            'Original image with border', 
            'Gray image', 
            'Blurred image', 
            'Canny filter', 
            'Dilated and eroded image', 
            'All contours', 
            'Top contours', 
            #'Clipped and transformed image'
        ]
        st.image(images, caption=captions)
        st.image(transformedImages, caption=[f'Document_{i}' for i in range(len(transformedImages))])
    return transformedImages
    #except:
        #st.info('Cannot process this image, change your thresholds.') 

    
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
    threshold2: int = st.slider('Threshold 2', 0, 255, 255, disabled=False)
    get_clipped_img(img, width, height, threshold1, threshold2)
