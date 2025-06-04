import numpy as np

def pad_image(img, kernel):
    """Thêm padding vào ảnh để tránh lỗi tràn khi thực hiện phép toán hình thái."""
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2  # Tính toán số pixel cần pad
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

def erode(img, kernel):
    """Thực hiện phép co (Erosion) để thu nhỏ vùng sáng."""
    padded_img = pad_image(img, kernel)  # Thêm padding vào ảnh
    result = np.zeros_like(img)  # Khởi tạo ảnh kết quả với giá trị 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]  # Lấy vùng con
            if np.array_equal(region * kernel, kernel):  # Kiểm tra nếu trùng khớp với kernel
                result[i, j] = 1  # Đặt giá trị pixel là 1
    return result

def dilate(img, kernel):
    """Thực hiện phép giãn (Dilation) để mở rộng vùng sáng."""
    padded_img = pad_image(img, kernel)  # Thêm padding vào ảnh
    result = np.zeros_like(img)  # Khởi tạo ảnh kết quả với giá trị 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]  # Lấy vùng con
            if np.any(region * kernel):  # Nếu có ít nhất một phần tử là 1
                result[i, j] = 1  # Đặt giá trị pixel là 1
    return result

def opening(img, kernel):
    """Phép mở: Erosion trước, sau đó Dilation (giúp loại bỏ nhiễu nhỏ)."""
    return dilate(erode(img, kernel), kernel)

def closing(img, kernel):
    """Phép đóng: Dilation trước, sau đó Erosion (giúp lấp đầy các lỗ hổng nhỏ)."""
    return erode(dilate(img, kernel), kernel)

def hit_or_miss(img, kernel):
    """Phép toán Hit-or-Miss để tìm các mẫu hình dạng cụ thể trong ảnh."""
    complement = 1 - img  # Lấy ảnh nền (background)
    
    # Xác định hai phần của kernel
    kernel_fg = (kernel == 1).astype(np.uint8)  # B1: foreground (các giá trị 1 trong kernel)
    kernel_bg = (kernel == -1).astype(np.uint8)  # B2: background (các giá trị -1 trong kernel)

    # Thực hiện phép co trên cả hai phần
    eroded_fg = erode(img, kernel_fg)  # Co ảnh với foreground
    eroded_bg = erode(complement, kernel_bg)  # Co ảnh với background
    
    # Lấy giao của hai ảnh co để tìm vùng khớp hoàn toàn
    return np.logical_and(eroded_fg, eroded_bg).astype(np.uint8)

def boundary_extraction(img, kernel):
    """Tách đường biên của vùng sáng trong ảnh."""
    return img - erode(img, kernel)  # Lấy phần ảnh ban đầu trừ đi ảnh bị co

def region_filling(img, kernel, seed):
    """Thuật toán lấp đầy vùng sáng dựa trên phép toán giãn (Dilation)."""
    result = np.zeros_like(img, dtype=np.uint8)  # Ảnh kết quả ban đầu (tất cả là 0)
    result[seed] = 1  # Đặt pixel seed ban đầu thành 1

    # Xác định vùng nền (background)
    background = 1 - img  # Đảm bảo chỉ mở rộng vào vùng nền
    
    while True:
        new_result = dilate(result, kernel) & background  # Giãn vùng seed nhưng giữ trong nền
        if np.array_equal(new_result, result):  # Nếu không có thay đổi, dừng lặp
            break
        result = new_result  # Cập nhật ảnh kết quả
    
    return result
