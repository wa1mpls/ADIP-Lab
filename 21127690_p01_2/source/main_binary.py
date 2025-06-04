import sys
import getopt
import cv2
import numpy as np
import os
import time
from morphological_operator import binary  # Import thư viện xử lý hình thái tự viết
# Import trực tiếp các hàm từ binary.py

def apply_manual(img, kernel, seed=(10, 10)):
    print("apply_manual() function called")  # Kiểm tra xem hàm được gọi chưa

    print("Creating operations dictionary...")

    operations = {}
    try:
        print("Adding Original...")
        operations["Original"] = img

        print("Adding Dilate...")
        operations["Dilate"] = binary.dilate(img, kernel)

        print("Adding Erode...")
        operations["Erode"] = binary.erode(img, kernel)

        print("Adding Open...")
        operations["Open"] = binary.opening(img, kernel)

        print("Adding Close...")
        operations["Close"] = binary.closing(img, kernel)

        print("Adding HitMiss...")
        operations["HitMiss"] = binary.hit_or_miss(img, kernel)

        print("Adding Boundary...")
        operations["Boundary"] = binary.boundary_extraction(img, kernel)

        print("Adding Fill...")
        operations["Fill"] = binary.region_filling(img, kernel, seed)

        print("Adding ConnectedComponents...")
        operations["ConnectedComponents"] = binary.connected_components(img)

        print("Adding ConvexHull...")
        operations["ConvexHull"] = binary.convex_hull(img)

        print("Adding Thinning...")
        operations["Thinning"] = binary.thinning(img)

        print("Adding Thickening...")
        operations["Thickening"] = binary.thickening(img, kernel)

        print("Adding Skeletonization...")
        operations["Skeletonization"] = binary.skeletonization(img)

        print("Adding Reconstruction...")
        operations["Reconstruction"] = binary.reconstruction(img, img, kernel)

        print("Adding Pruning...")
        operations["Pruning"] = binary.pruning(img)

    except Exception as e:
        print(f"Error when adding to operations: {e}")
        sys.exit(1)

    print("Operations dictionary created:", list(operations.keys()))
    return operations

def apply_opencv(img, kernel, seed=(10, 10)):
    """Áp dụng các phép toán hình thái bằng OpenCV."""
    # OpenCV có hỗ trợ một số phép toán như Dilate, Erode, Open, Close, HitMiss
    # ConnectedComponents và Skeletonization cũng được hỗ trợ trong OpenCV
    return {
        "Original": img,
        "Dilate (OpenCV)": cv2.dilate(img, kernel),  # Giãn nở dùng OpenCV
        "Erode (OpenCV)": cv2.erode(img, kernel),  # Co lại dùng OpenCV
        "Open (OpenCV)": cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),  # Mở
        "Close (OpenCV)": cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),  # Đóng
        "HitMiss (OpenCV)": cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel),  # Hit-or-Miss
        "Boundary (OpenCV)": cv2.dilate(img, kernel) - img,  # Biên = (Giãn nở - Ảnh gốc)
        "Fill (OpenCV)": cv2.floodFill(img.copy(), None, seed, 255)[1],  # Lấp vùng bằng floodFill
        "ConnectedComponents (OpenCV)": cv2.connectedComponents(img)[1],  # Tách thành phần liên thông
        # OpenCV không có hàm trực tiếp cho Skeletonization, nhưng có thể dùng thinning của OpenCV
        "Skeletonization (OpenCV)": cv2.ximgproc.thinning(img * 255) // 255  # Làm mỏng để gần giống bộ xương
    }

def operator(in_file, out_file, mor_op, mode, wait_key_time=0):
    print(f"operator() called with: in_file={in_file}, out_file={out_file}, mor_op={mor_op}, mode={mode}")

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
    _, img = cv2.threshold(img_origin, 128, 1, cv2.THRESH_BINARY)

    # Kernel 3x3 dùng cho phép toán hình thái
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # Điểm seed mặc định cho Fill và Reconstruction
    seed = (10, 10)

    # Bắt đầu tính thời gian thực thi
    start_time = time.time()

   
    # Chọn chế độ thực hiện
    print(f"Mode received: {mode}")
    if mode == "manual":
        print("Calling apply_manual()...")
        operations = apply_manual(img, kernel, seed)  
        if operations is None:
            print("Error: apply_manual() returned None")
            sys.exit(1)
        print("Operations dictionary:", list(operations.keys()))
        print("apply_manual() executed successfully.")
        method = "Manual (Custom)"    
    elif mode == "opencv":
        operations = apply_opencv(img, kernel, seed)  # Dùng OpenCV
        method = "OpenCV"
    else:
        print("Error: Invalid mode. Choose 'manual' or 'opencv'.")
        sys.exit(1) 

    

    exec_time = time.time() - start_time  # Thời gian thực thi

    
 

    # Nếu chỉ thực hiện một phép toán cụ thể
    if mor_op:
        if mor_op in operations:
            print(f"Executing {mor_op} using {method}")
            img_out = operations[mor_op]
            # Chuẩn hóa ảnh đầu ra để hiển thị (nếu cần)
            if img_out is None or img_out.size == 0:
                print(f"Error: '{mor_op}' returned an empty result.")
                sys.exit(1)
            if img_out.dtype != np.uint8 or img_out.max() > 1:
                img_out_display = (img_out / img_out.max() * 255).astype(np.uint8)
            else:
                img_out_display = img_out * 255
            cv2.imshow(f"Result: {mor_op} - {method}", img_out_display)  # Hiển thị kết quả
            cv2.imwrite(out_file, img_out_display)  # Lưu ảnh kết quả
            cv2.waitKey(wait_key_time)
            print(f"Output saved to {out_file}")
            print(f"Time Complexity ({method}): {exec_time:.6f} seconds")
        else:
            print(f"Error: Unknown morphological operation '{mor_op}'")
    else:
        # Nếu không chọn phép toán cụ thể, hiển thị tất cả kết quả trên một ảnh
        rows, cols = 4, 4  # Tăng lưới để chứa nhiều phép toán hơn (16 slot)
        images = list(operations.values())  # Lấy danh sách ảnh kết quả
        labels = list(operations.keys())  # Nhãn cho từng ảnh

        h, w = images[0].shape  # Kích thước ảnh
        label_height = 30  # Kích thước vùng hiển thị nhãn
        grid_img = np.ones((h * rows + label_height * rows, w * cols), dtype=np.uint8) * 255  # Tạo ảnh nền trắng

        # Ghép ảnh vào lưới
        for idx, (label, img) in enumerate(zip(labels, images)):
            if idx >= rows * cols:  # Giới hạn số lượng ảnh hiển thị
                break
            row, col = divmod(idx, cols)  # Xác định vị trí trong lưới
            y_start = row * (h + label_height)
            y_end = y_start + h
            x_start = col * w
            x_end = x_start + w

            # Chuẩn hóa ảnh để hiển thị (nếu cần)
            if img.dtype != np.uint8 or img.max() > 1:
                img_display = (img / img.max() * 255).astype(np.uint8)
            else:
                img_display = img * 255

            grid_img[y_start:y_end, x_start:x_end] = img_display  # Đưa ảnh vào grid
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