import sys
import getopt
import cv2
import numpy as np
import os
import time
from morphological_operator import binary  # Import thư viện xử lý hình thái tự viết

def apply_manual(img, kernel):
    """Áp dụng các phép toán hình thái bằng thuật toán tự viết."""
    return {
        "Original": img,  # Ảnh gốc
        "Dilate": binary.dilate(img, kernel),  # Giãn nở (Dilation)
        "Erode": binary.erode(img, kernel),  # Co lại (Erosion)
        "Open": binary.opening(img, kernel),  # Mở (Opening)
        "Close": binary.closing(img, kernel),  # Đóng (Closing)
        "HitMiss": binary.hit_or_miss(img, kernel),  # Hit-or-Miss
        "Boundary": binary.boundary_extraction(img, kernel),  # Trích xuất biên
        "Fill": binary.region_filling(img, kernel, (10, 10))  # Lấp vùng
    }

def apply_opencv(img, kernel):
    """Áp dụng các phép toán hình thái bằng OpenCV."""
    return {
        "Original": img,
        "Dilate (OpenCV)": cv2.dilate(img, kernel),  # Giãn nở dùng OpenCV
        "Erode (OpenCV)": cv2.erode(img, kernel),  # Co lại dùng OpenCV
        "Open (OpenCV)": cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),  # Mở
        "Close (OpenCV)": cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),  # Đóng
        "HitMiss (OpenCV)": cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel),  # Hit-or-Miss
        "Boundary (OpenCV)": cv2.dilate(img, kernel) - img,  # Biên = (Giãn nở - Ảnh gốc)
        "Fill (OpenCV)": cv2.floodFill(img.copy(), None, (10, 10), 255)[1]  # Lấp vùng bằng floodFill
    }

def operator(in_file, out_file, mor_op, mode, wait_key_time=0):
    """Thực hiện phép toán hình thái trên ảnh."""
    
    # Kiểm tra xem tệp ảnh có tồn tại không
    if not os.path.exists(in_file):
        print(f"Error: Input file '{in_file}' not found.")
        sys.exit(1)

    # Đọc ảnh đầu vào ở chế độ grayscale
    img_origin = cv2.imread(in_file, 0)
    if img_origin is None:
        print(f"Error: Unable to read image file '{in_file}'. Check file format and path.")
        sys.exit(1)

    # Chuyển ảnh về nhị phân bằng threshold
    #_, img = cv2.threshold(img_origin, 128, 1, cv2.THRESH_BINARY_INV if np.mean(img_origin) > 128 else cv2.THRESH_BINARY)
    _, img = cv2.threshold(img_origin, 128, 1, cv2.THRESH_BINARY)

    

    # Kernel 3x3 dùng cho phép toán hình thái
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # Bắt đầu tính thời gian thực thi
    start_time = time.time()

    # Chọn chế độ thực hiện
    if mode == "manual":
        operations = apply_manual(img, kernel)  # Dùng thuật toán tự viết
        method = "Manual (Custom)"
    elif mode == "opencv":
        operations = apply_opencv(img, kernel)  # Dùng OpenCV
        method = "OpenCV"
    else:
        print("Error: Invalid mode. Choose 'manual' or 'opencv'.")
        sys.exit(1)

    exec_time = time.time() - start_time  # Thời gian thực thi

    # Nếu chỉ thực hiện một phép toán cụ thể
    if mor_op:
        if mor_op in operations:
            img_out = operations[mor_op]
            cv2.imshow(f"Result: {mor_op} - {method}", img_out * 255)  # Hiển thị kết quả
            cv2.imwrite(out_file, img_out * 255)  # Lưu ảnh kết quả
            cv2.waitKey(wait_key_time)
            print(f"Output saved to {out_file}")
            print(f"Time Complexity ({method}): {exec_time:.6f} seconds")
        else:
            print(f"Error: Unknown morphological operation '{mor_op}'")
    else:
        # Nếu không chọn phép toán cụ thể, hiển thị tất cả kết quả trên một ảnh
        rows, cols = 2, 4  # Xếp lưới 2 hàng 4 cột
        images = list(operations.values())  # Lấy danh sách ảnh kết quả
        labels = list(operations.keys())  # Nhãn cho từng ảnh

        h, w = images[0].shape  # Kích thước ảnh
        label_height = 30  # Kích thước vùng hiển thị nhãn
        grid_img = np.ones((h * rows + label_height * rows, w * cols), dtype=np.uint8) * 255  # Tạo ảnh nền trắng

        # Ghép ảnh vào lưới
        for idx, (label, img) in enumerate(zip(labels, images)):
            row, col = divmod(idx, cols)  # Xác định vị trí trong lưới
            y_start = row * (h + label_height)
            y_end = y_start + h
            x_start = col * w
            x_end = x_start + w

            grid_img[y_start:y_end, x_start:x_end] = img * 255  # Đưa ảnh vào grid
            cv2.putText(grid_img, label, (x_start + 5, y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)  # Ghi nhãn

        cv2.imshow(f"All Morphological Operations - {method}", grid_img)  # Hiển thị ảnh tổng hợp
        cv2.imwrite(out_file, grid_img)  # Lưu ảnh tổng hợp
        cv2.waitKey(wait_key_time)
        print(f"All operations saved to {out_file}")
        print(f"Execution Time ({method}): {exec_time:.6f} seconds")

def main(argv):
    """Xử lý đầu vào từ dòng lệnh."""
    input_file = ''
    output_file = ''
    mor_op = ''
    mode = 'manual'  # Mặc định dùng thuật toán thủ công
    wait_key_time = 0

    description = 'Usage: main.py -i <input_file> -o <output_file> [-p <morph_operator>] -m <mode> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:m:t:", ["in_file=", "out_file=", "mor_operator=", "mode=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-m", "--mode"):
            mode = arg.lower()  # Chuyển về chữ thường
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    if not input_file or not output_file:
        print("Error: Missing required arguments.")
        print(description)
        sys.exit(1)

    operator(input_file, output_file, mor_op, mode, wait_key_time)

if __name__ == "__main__":
    main(sys.argv[1:])
