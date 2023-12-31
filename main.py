import streamlit as st
import numpy as np
import scipy.spatial.distance
import cv2
import time


st.set_page_config(page_title='Document scanner demo', page_icon=':document:')


file = None

def add_border(img: np.ndarray, border_size: int = 50, value: int = 255) -> np.ndarray:
    border = [border_size for i in range(4)]
    value = [value for i in range(3)]
    return cv2.copyMakeBorder(img, *border, cv2.BORDER_CONSTANT, value=value)

@st.cache_data
def enhance_contrast(img:np.ndarray) -> np.ndarray:
    lab: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe: np.ndarray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl: np.ndarray = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg: np.ndarray = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img: np.ndarray = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    img: np.ndarray = cv2.resize(img, (width, height))
    return img
    

@st.cache_data
def to_grayscale(img: np.ndarray) -> np.ndarray:
    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    

@st.cache_data
def add_gaussian_blur(img: np.ndarray) -> np.ndarray:
    img: np.ndarray = cv2.GaussianBlur(img, (5, 5), 1)
    return img
    

@st.cache_data
def apply_Canny_filter(img: np.ndarray, threshold1: int = 50, threshold2: int = 255, aperture_size: int = 3) -> np.ndarray:
    img: np.ndarray = cv2.Canny(
        image=img,
        threshold1=threshold1,
        threshold2=threshold2,
        apertureSize=3,
        L2gradient=True
    )
    return img
    

@st.cache_data
def dilate_and_erode(img: np.ndarray, kernel: np.ndarray = np.ones((3, 3)), d_iter: int = 1, e_iter: int = 1) -> np.ndarray:
    imgDial: np.ndarray = cv2.dilate(img, kernel, iterations=d_iter)
    imgThreshold: np.ndarray = cv2.erode(imgDial, kernel, iterations=e_iter)
    return imgThreshold
    

@st.cache_data
def get_all_contours(img: np.ndarray, imgThreshold: np.ndarray) -> tuple[tuple[np.ndarray], np.ndarray]:
    imgContours: np.ndarray = img.copy()
    contours: tuple[np.ndarray, None] = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: np.ndarray = contours[0]
    allContours: np.ndarray = cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    return contours, allContours
    

@st.cache_data
def get_top_contours(img: np.ndarray, contours: np.ndarray, num_corners: set[int] = set([4])) -> tuple[np.ndarray, np.ndarray]:
    top_contours: list[np.ndarray] = []
    imgContours: np.ndarray = img.copy()
    contour_area_threshold: float = (img.shape[0] * img.shape[1]) / 30.0
    st.info(contour_area_threshold)
    for contour in contours:
        area: float = cv2.contourArea(contour)
        if area > contour_area_threshold:
            peri = cv2.arcLength(contour, True)
            approx: np.ndarray = cv2.approxPolyDP(contour, 0.02 * peri, True)
            #st.info(f'Len: {len(approx)}')
            if len(approx) in num_corners:
                top_contours.append(approx)
    top_contours = np.array(top_contours)
    imgContours: np.ndarray = cv2.drawContours(imgContours, top_contours, -1, (0, 255, 0), 10)
    return top_contours, imgContours
    

@st.cache_data
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
    

@st.cache_data
def transform_with_ratio(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape

    #image center
    u0: float = (cols) / 2.0
    v0: float = (rows) / 2.0

    #detected corners on the original image
    p: np.ndarray = np.squeeze(pts)

    #widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])

    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])

    w = np.max(np.array([w1, w2]))
    h = np.max(np.array([h1, h2]))

    #visible aspect ratio
    ar_vis = float(w) / float(h)

    #make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    #calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = np.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

    A = np.array([[f, 0, u0], [0, f, v0],[0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    #calculate the real aspect ratio
    ar_real = np.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(p).astype('float32')
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

    #project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (W, H))
    return dst


def transform(img: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    matrix: np.ndarray = cv2.getPerspectiveTransform(pts1, pts2)
    result: np.ndarray = cv2.warpPerspective(img, matrix, (width, height))
    return result
    

@st.cache_data
def get_clipped_img(
    img: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 100,
    kernel: tuple[int, int] = (3, 3),
    num_corners: set[int] = set([4]),
    verbose: bool = True, 
    enhance: bool = True
) -> np.ndarray:
    images = [img]
    captions = ['Original image']
    #imgBorder = add_border(img)
    #resizedImg: np.ndarray = resize(img, width, height)
    if enhance:
        enhancedImg = enhance_contrast(img)
        images.append(enhancedImg)
        captions.append('Enhanced contrast')
        grayImg: np.ndarray = to_grayscale(enhancedImg)
    else:
        grayImg: np.ndarray = to_grayscale(img)
    images.append(grayImg)
    captions.append('Grayscale image') 
    blurredImg: np.ndarray = add_gaussian_blur(grayImg)
    images.append(blurredImg)
    captions.append('Blurred image')
    CannyImg: np.ndarray = apply_Canny_filter(blurredImg, threshold1, threshold2)
    images.append(CannyImg)
    captions.append('Canny filter')
    deImg: np.ndarray = dilate_and_erode(CannyImg, kernel)
    images.append(deImg)
    captions.append('Dilated and eroded image')
    allContours: tuple[tuple[np.ndarray], np.ndarray]  = get_all_contours(img, deImg)
    contoursImg: np.ndarray = allContours[1]
    allContours: tuple[np.ndarray] = allContours[0]
    images.append(contoursImg)
    captions.append('All contours')
    topContours: tuple[np.ndarray, np.ndarray] = get_top_contours(img, allContours, num_corners)
    imgContours: np.ndarray = topContours[1]
    topContours: np.ndarray = topContours[0]
    images.append(imgContours)
    captions.append('Top contours')
    pts1: np.ndarray = reorder(topContours)
    try:
        transformedImages: list[np.ndarray] = [transform_with_ratio(img, p) for p in pts1]
    except:
        st.warning(f'Something is wrong with image transformation function')
    if verbose:
        st.image(images, caption=captions)
    try:
        st.image(transformedImages, caption=[f'Document_{i}' for i in range(len(transformedImages))])
        return transformedImgs
    except:
        pass

    
'''
# Hello!

This toy website allows you to interact with image document scanner service.

Just upload your image with document below!
'''

file = st.file_uploader('Upload your documents', accept_multiple_files=False)
if file:
    file_bytes: np.ndarray = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img: np.ndarray = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
else:
    img: np.ndarray = cv2.cvtColor(cv2.imread('Без названия.png'), cv2.COLOR_BGR2RGB)
    
threshold1: int = st.slider('Threshold 1', 0, 255, 50, disabled=False)
threshold2: int = st.slider('Threshold 2', 0, 255, 255, disabled=False)
num_corners: tuple[int, int] = st.slider('Number of corners', 4, 8, (4, 5))
num_corners: set[int] = set(list(range(num_corners[0], num_corners[1] + 1)))
verbose: bool = st.checkbox('Verbose', True)
enhance: bool = st.checkbox('Enhance contrast', True)
kernel: np.ndarray = np.ones(st.selectbox('Dilate and eroded kernel size', [(3, 3), (5, 5), (7, 7)]))
#st.info(num_corners)
documents: list[np.ndarray] = get_clipped_img(img, threshold1, threshold2, kernel, num_corners, verbose, enhance)

#if documents is not None:
#    st.download_button(
#        'Download images', 
#        data=documents,
#        mime='image/png', 
#        type='primary'
#    )
