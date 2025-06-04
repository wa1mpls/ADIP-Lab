import numpy as np

def pad_image_grayscale(img, kernel, pad_value):
    """Thêm padding với giá trị tùy theo phép toán (0 cho dilation, 255 cho erosion)."""
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=pad_value)

def grayscale_dilation(img, kernel):
    """Grayscale Dilation: (f ⊕ b)(s,t) = max{f(s-x, t-y) + b(x,y)}"""
    padded_img = pad_image_grayscale(img, kernel, 0)  # Pad với 0
    result = np.zeros_like(img, dtype=np.uint8)
    kh, kw = kernel.shape

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kh, j:j+kw]
            result[i, j] = np.clip(np.max(region + kernel), 0, 255)  # Giữ giá trị trong [0, 255]

    return result



def grayscale_erosion(img, kernel):
    """Grayscale Erosion: (f ⊖ b)(s,t) = min{f(s+x, t+y) - b(x,y)}"""
    padded_img = pad_image_grayscale(img, kernel, 255)  # Pad với 255 cho erosion
    result = np.zeros_like(img, dtype=np.uint8)
    kh, kw = kernel.shape

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kh, j:j+kw]
            result[i, j] = np.clip(np.min(region - kernel), 0, 255)  # Giữ giá trị trong [0, 255]

    return result



def grayscale_opening(img, kernel):
    """Grayscale Opening: f ∘ b = (f ⊖ b) ⊕ b"""
    return grayscale_dilation(grayscale_erosion(img, kernel), kernel)

def grayscale_closing(img, kernel):
    """Grayscale Closing: f • b = (f ⊕ b) ⊖ b"""
    return grayscale_erosion(grayscale_dilation(img, kernel), kernel)

def grayscale_smoothing(img, kernel):
    """Grayscale Smoothing: Áp dụng Opening rồi Closing để làm mịn ảnh."""
    return grayscale_closing(grayscale_opening(img, kernel), kernel)

def grayscale_morphology_gradient(img, kernel):
    """Grayscale Morphology Gradient: (f ⊕ b) - (f ⊖ b)"""
    dilated = grayscale_dilation(img, kernel)
    eroded = grayscale_erosion(img, kernel)
    return np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 255).astype(np.uint8)

def top_hat(img, kernel):
    """Top-hat Transformation: f - (f ∘ b)"""
    opened = grayscale_opening(img, kernel)
    return np.clip(img.astype(np.int16) - opened.astype(np.int16), 0, 255).astype(np.uint8)

def textural_segmentation(img, kernel):
    """Textural Segmentation: Dùng Top-hat để phân đoạn kết cấu"""
    return top_hat(img, kernel)

def granulometry(img, sizes):
    """Granulometry: Tính tổng giá trị pixel sau Opening với các kích thước kernel khác nhau"""
    granulometry_result = []
    for size in sizes:
        kernel = np.ones((size, size), dtype=np.uint8)
        opened = grayscale_opening(img, kernel)
        granulometry_result.append(np.sum(opened))
    return granulometry_result

def reconstruction(marker, mask, kernel, max_iter=100):
    """Morphological Reconstruction by Dilation: R_g^D(f)"""
    prev_marker = np.zeros_like(marker)
    iter_count = 0

    while not np.array_equal(marker, prev_marker):
        prev_marker = marker.copy()
        marker = np.minimum(grayscale_dilation(marker, kernel), mask)
        iter_count += 1
        if iter_count >= max_iter:
            print("Warning: Reconstruction reached max iterations!")
            break

    return np.clip(marker, 0, 255).astype(np.uint8)

def create_structuring_element(size):
    """Tạo phần tử cấu trúc hình vuông"""
    return np.ones((size, size), dtype=np.uint8)
