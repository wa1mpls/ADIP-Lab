import numpy as np

def pad_image(img, kernel):
    """Thêm padding vào ảnh để tránh lỗi tràn khi thực hiện phép toán hình thái."""
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

def erode(img, kernel):
    """Thực hiện phép co (Erosion) để thu nhỏ vùng sáng."""
    padded_img = pad_image(img, kernel)
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            if np.array_equal(region * kernel, kernel):
                result[i, j] = 1
    return result

def dilate(img, kernel):
    """Thực hiện phép giãn (Dilation) để mở rộng vùng sáng."""
    padded_img = pad_image(img, kernel)
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            if np.any(region * kernel):
                result[i, j] = 1
    return result

def opening(img, kernel):
    """Phép mở: Erosion trước, sau đó Dilation (giúp loại bỏ nhiễu nhỏ)."""
    return dilate(erode(img, kernel), kernel)

def closing(img, kernel):
    """Phép đóng: Dilation trước, sau đó Erosion (giúp lấp đầy các lỗ hổng nhỏ)."""
    return erode(dilate(img, kernel), kernel)

def hit_or_miss(img, kernel):
    """Phép toán Hit-or-Miss để tìm các mẫu hình dạng cụ thể trong ảnh."""
    complement = 1 - img
    kernel_fg = (kernel == 1).astype(np.uint8)
    kernel_bg = (kernel == -1).astype(np.uint8)

    eroded_fg = erode(img, kernel_fg)
    eroded_bg = erode(complement, kernel_bg)
    
    return np.logical_and(eroded_fg, eroded_bg).astype(np.uint8)

def boundary_extraction(img, kernel):
    """Tách đường biên của vùng sáng trong ảnh."""
    return img - erode(img, kernel)

def region_filling(img, kernel, seed):
    """Thuật toán lấp đầy vùng sáng dựa trên phép toán giãn (Dilation)."""
    result = np.zeros_like(img, dtype=np.uint8)
    result[seed] = 1

    background = 1 - img
    
    while True:
        new_result = dilate(result, kernel) & background
        if np.array_equal(new_result, result):
            break
        result = new_result
    
    return result

# 7 Thuật toán bổ sung

from collections import deque

from collections import deque
import numpy as np

def connected_components(img):
    """
    Tách các thành phần liên thông trong ảnh nhị phân bằng thuật toán BFS.
    
    Parameters:
        img (numpy.ndarray): Ảnh nhị phân đầu vào (0: nền, 1: vật thể).

    Returns:
        numpy.ndarray: Ảnh đầu ra với các thành phần liên thông được gán nhãn khác nhau.
    """
    h, w = img.shape  # Lấy kích thước của ảnh
    labels = np.zeros((h, w), dtype=np.int32)  # Tạo ma trận nhãn, ban đầu gán tất cả bằng 0
    label = 1  # Khởi tạo nhãn đầu tiên
    
    # Định nghĩa các hướng di chuyển trong lân cận 8 điểm
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Duyệt qua từng pixel trong ảnh
    for i in range(h):
        for j in range(w):
            # Nếu pixel là một phần của vật thể (giá trị 1) và chưa được gán nhãn
            if img[i, j] == 1 and labels[i, j] == 0:
                queue = deque([(i, j)])  # Tạo hàng đợi BFS
                while queue:
                    x, y = queue.popleft()  # Lấy điểm đầu tiên trong queue
                    
                    # Nếu pixel này chưa được gán nhãn, thì gán nhãn cho nó
                    if labels[x, y] == 0:
                        labels[x, y] = label
                        
                        # Kiểm tra các pixel lân cận
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy  # Tính tọa độ mới
                            # Nếu tọa độ hợp lệ, pixel thuộc vật thể và chưa được gán nhãn
                            if 0 <= nx < h and 0 <= ny < w and img[nx, ny] == 1 and labels[nx, ny] == 0:
                                queue.append((nx, ny))  # Đưa vào hàng đợi để tiếp tục duyệt
                
                label += 1  # Tăng nhãn cho thành phần liên thông mới

    return labels  # Trả về ảnh với các thành phần liên thông đã được đánh nhãn



def convex_hull(img):
    """
    Tính toán bao lồi của một vùng sáng trong ảnh bằng thuật toán Jarvis March (Gift Wrapping).

    Parameters:
        img (numpy.ndarray): Ảnh nhị phân đầu vào (0: nền, 1: vùng sáng).

    Returns:
        numpy.ndarray: Ảnh nhị phân có đường bao lồi.
    """
    
    # Lấy danh sách tọa độ các điểm có giá trị 1 (điểm thuộc vùng sáng)
    points = np.argwhere(img == 1)
    
    # Nếu ảnh không có điểm sáng nào, trả về ảnh ban đầu
    if len(points) == 0:
        return img  

    def cross_product(o, a, b):
        """
        Tính tích có hướng giữa hai vector oa và ob.
        Giá trị âm -> b nằm bên phải oa (ngược chiều kim đồng hồ).
        Giá trị dương -> b nằm bên trái oa (thuận chiều kim đồng hồ).
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Tìm điểm có tọa độ (y, x) nhỏ nhất -> đây là điểm xuất phát của convex hull
    start = points[np.argmin(points[:, 1])]  # Chọn điểm có x nhỏ nhất (nếu trùng thì chọn y nhỏ nhất)
    hull = [start]  # Danh sách chứa các điểm của convex hull

    while True:
        candidate = None  # Điểm kế tiếp trong convex hull
        for p in points:
            if np.array_equal(p, hull[-1]):  # Bỏ qua chính điểm hiện tại
                continue
            
            if candidate is None or cross_product(hull[-1], candidate, p) < 0:
                candidate = p  # Chọn điểm xa nhất theo chiều ngược kim đồng hồ
            
            elif cross_product(hull[-1], candidate, p) == 0:
                # Nếu ba điểm thẳng hàng, chọn điểm xa hơn
                if np.linalg.norm(p - hull[-1]) > np.linalg.norm(candidate - hull[-1]):
                    candidate = p
        
        if np.array_equal(candidate, start):  # Nếu quay lại điểm đầu tiên, thuật toán kết thúc
            break
        
        hull.append(candidate)  # Thêm điểm mới vào convex hull

    # Tạo ảnh nhị phân mới chứa bao lồi
    hull_img = np.zeros_like(img)
    for p in hull:
        hull_img[p[0], p[1]] = 1  # Đánh dấu các điểm thuộc đường bao lồi

    return hull_img



def thinning(img):
    """Làm mỏng ảnh bằng thuật toán Zhang-Suen."""
    def count_transitions(P):
        """Đếm số lần chuyển từ 0 -> 1 theo thứ tự vòng."""
        P = [P[1,2], P[2,2], P[2,1], P[2,0], P[1,0], P[0,0], P[0,1], P[0,2], P[1,2]]
        return sum((P[i] == 0 and P[i+1] == 1) for i in range(8))

    def step(img, pass_num):
        markers = np.zeros_like(img)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                P = img[i-1:i+2, j-1:j+2]
                if img[i, j] == 1:
                    neighbors = np.sum(P) - 1
                    transitions = count_transitions(P)
                    cond1 = 2 <= neighbors <= 6
                    cond2 = transitions == 1
                    cond3 = P[0,1] * P[1,2] * P[2,1] == 0 if pass_num == 0 else P[1,2] * P[2,1] * P[1,0] == 0
                    if cond1 and cond2 and cond3:
                        markers[i, j] = 1
        img[markers == 1] = 0

    prev = np.zeros_like(img)
    while not np.array_equal(img, prev):
        prev = img.copy()
        step(img, 0)
        step(img, 1)
    
    return img

def thickening(img, kernel):
    """Làm dày ảnh bằng dilation có điều kiện."""
    complement = np.logical_not(img)
    new_img = dilate(img, kernel)
    return np.logical_and(new_img, complement).astype(np.uint8)


def skeletonization(img):
    """Tạo bộ xương của đối tượng bằng phương pháp lặp erosion."""
    skeleton = np.zeros_like(img)
    temp = img.copy()
    while np.any(temp):
        eroded = erode(temp, np.ones((3, 3), np.uint8))
        skeleton = np.logical_or(skeleton, temp - eroded).astype(np.uint8)
        temp = eroded
    return skeleton


def reconstruction(marker, mask, kernel):
    """Tái tạo ảnh từ một marker bằng phương pháp giãn nở có giới hạn."""
    while True:
        next_marker = np.minimum(dilate(marker, kernel), mask)
        if np.array_equal(next_marker, marker):
            break
        marker = next_marker
    return marker


def pruning(skeleton, iterations=1):
    """Cắt tỉa ảnh bộ xương bằng cách loại bỏ điểm cuối."""
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        endpoints = np.logical_and(skeleton, np.sum(erode(skeleton, kernel), axis=(0, 1)) == 1)
        skeleton = np.logical_and(skeleton, np.logical_not(endpoints)).astype(np.uint8)
    return skeleton

