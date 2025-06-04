import sys
import getopt
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from morphological_operator.grayscale import  pad_image_grayscale, grayscale_dilation, grayscale_erosion, grayscale_opening, grayscale_closing, grayscale_smoothing, grayscale_morphology_gradient, top_hat, textural_segmentation, granulometry, reconstruction


def apply_manual(img, kernel, seed=(10, 10)):
    """Áp dụng các phép toán hình thái grayscale bằng thuật toán tự viết."""
    print("apply_manual() function called")  # Kiểm tra xem hàm được gọi chưa

    print("Creating operations dictionary...")
    operations = {}
    sizes = [3, 5, 7]  # Các kích thước kernel cho Granulometry

    try:
        print("Adding Original...")
        operations["Original"] = img

        print("Adding Dilate...")
        operations["Dilate"] = grayscale_dilation(img, kernel)

        print("Adding Erode...")
        operations["Erode"] = grayscale_erosion(img, kernel)


        print("Adding Open...")
        operations["Open"] = grayscale_opening(img, kernel)

        print("Adding Close...")
        operations["Close"] = grayscale_closing(img, kernel)

        print("Adding Smoothing...")
        operations["Smoothing"] = grayscale_smoothing(img, kernel)

        print("Adding Gradient...")
        operations["Gradient"] = grayscale_morphology_gradient(img, kernel)

        print("Adding TopHat...")
        operations["TopHat"] = top_hat(img, kernel)

        print("Adding TexturalSegmentation...")
        operations["TexturalSegmentation"] = textural_segmentation(img, kernel)

        print("Adding Granulometry...")
        operations["Granulometry"] = granulometry(img, sizes)  # Trả về list, cần xử lý riêng khi hiển thị

        print("Adding Reconstruction...")
        operations["Reconstruction"] = reconstruction(img, img, kernel)  # Dùng img làm marker và mask

    except Exception as e:
        print(f"Error when adding to operations: {e}")
        sys.exit(1)

    print("Operations dictionary created:", list(operations.keys()))
    return operations

def apply_opencv(img, kernel, seed=(10, 10)):
    """Áp dụng các phép toán hình thái grayscale bằng OpenCV."""
    return {
        "Original": img,
        "Dilate": cv2.dilate(img, kernel, iterations=1),
        "Erode": cv2.erode(img, kernel, iterations=1),
        "Open": cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),
        "Close": cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),
        "Gradient": cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel),
        "TopHat": cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel),
    }

def operator(in_file, out_file, mor_op, mode, wait_key_time=5000):
    """Thực hiện phép toán hình thái trên ảnh grayscale."""
    print(f"Đang xử lý file đầu vào: {in_file}")
    
    if not os.path.exists(in_file):
        print(f"Error: Input file '{in_file}' not found.")
        sys.exit(1)

    img = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image file '{in_file}'. Check file format and path.")
        sys.exit(1)
    print(f"Đã đọc ảnh: {img.shape}")

    # Chuẩn hóa ảnh về [0, 1] để phù hợp với tính toán
    img = img.astype(np.uint8)  

    kernel = np.ones((3, 3), dtype=np.uint8)
    seed = (10, 10)

    start_time = time.time()

    if mode == "manual":
        operations = apply_manual(img, kernel, seed)
        method = "Manual (Custom)"
    elif mode == "opencv":
        operations = apply_opencv(img, kernel, seed)
        method = "OpenCV"
    else:
        print("Error: Invalid mode. Choose 'manual' or 'opencv'.")
        sys.exit(1)

    exec_time = time.time() - start_time
    mor_op = mor_op.capitalize() if mor_op else ""
    print(f"Danh sách phép toán khả dụng: {list(operations.keys())}")

    if mor_op:
        if mor_op in operations:
            result = operations[mor_op]
            if isinstance(result, list):  # Đặc biệt cho Granulometry
                print(f"Granulometry results for sizes [3, 5, 7]: {result}")
                # Không hiển thị ảnh cho Granulometry, chỉ lưu hoặc in kết quả
                np.savetxt(out_file.replace('.png', '.txt'), result, fmt='%.6f')
                print(f"Granulometry results saved to {out_file.replace('.png', '.txt')}")
            else:
                # Chuẩn hóa kết quả để hiển thị
                result_display = (result - result.min()) / (result.max() - result.min()) * 255
                result_display = result_display.astype(np.uint8)
                cv2.imshow(f"Result: {mor_op} - {method}", result_display)
                cv2.imwrite(out_file, result_display)
                cv2.waitKey(wait_key_time)
                cv2.destroyAllWindows()
                print(f"Output saved to {out_file}")
            print(f"Time Complexity ({method}): {exec_time:.6f} seconds")
        else:
            print(f"Error: Unknown morphological operation '{mor_op}'")
            print("Available operations:", list(operations.keys()))
            sys.exit(1)
    else:
        rows, cols = 2, 5  # Điều chỉnh lưới cho số lượng phép toán phù hợp
        images = [op for key, op in operations.items() if key != "Granulometry"]
        labels = [key for key in operations.keys() if key != "Granulometry"]

        h, w = images[0].shape
        label_height = 30
        grid_img = np.ones((h * rows + label_height * rows, w * cols), dtype=np.uint8) * 255

        for idx, (label, img) in enumerate(zip(labels, images)):
            if idx >= rows * cols:
                break
            row, col = divmod(idx, cols)
            y_start = row * (h + label_height)
            y_end = y_start + h
            x_start = col * w
            x_end = x_start + w

            if img.max() - img.min() == 0:
                img_display = np.zeros_like(img, dtype=np.uint8)  # Gán ảnh về 0 nếu không có sự thay đổi
            else:
                img_display = (img - img.min()) / (img.max() - img.min()) * 255
                img_display = img_display.astype(np.uint8)

            img_display = img_display.astype(np.uint8)
            grid_img[y_start:y_end, x_start:x_end] = img_display
            cv2.putText(grid_img, label, (x_start + 5, y_end + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

        cv2.imshow(f"All Morphological Operations - {method}", grid_img)
        cv2.imwrite(out_file, grid_img)
        cv2.waitKey(wait_key_time)
        cv2.destroyAllWindows()
        print(f"All operations saved to {out_file}")
        import matplotlib.pyplot as plt

        if "Granulometry" in operations:
            granulometry_result = operations["Granulometry"]
            sizes = list(range(1, len(granulometry_result) + 1))  # Tạo danh sách kích thước SE

            # Vẽ biểu đồ Granulometry
            plt.figure(figsize=(8, 5))
            plt.plot(sizes, granulometry_result, marker='o', linestyle='-', color='b', label="Granulometry Profile")
            
            # Thêm tiêu đề và nhãn
            plt.xlabel("Structuring Element Size")
            plt.ylabel("Sum of Pixels After Opening")
            plt.title("Granulometry Analysis")
            plt.legend()
            plt.grid(True)

            # Lưu file hình ảnh
            img_filename = out_file.replace('.png', '_granulometry.png')
            plt.savefig(img_filename, dpi=300)  # Lưu với độ phân giải 300 dpi
            print(f"Granulometry plot saved to {img_filename}")

            # Hiển thị biểu đồ
            plt.show()

        print(f"Execution Time ({method}): {exec_time:.6f} seconds")


def main(argv):
    """Xử lý đầu vào từ dòng lệnh."""
    input_file = ''
    output_file = ''
    mor_op = ''
    mode = 'manual'
    wait_key_time = 5000

    description = 'Usage: main_grayscale.py -i <input_file> -o <output_file> [-p <morph_operator>] -m <mode> -t <wait_key_time>'

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
            mode = arg.lower()
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    if not input_file or not output_file:
        print("Error: Missing required arguments.")
        print(description)
        sys.exit(1)

    operator(input_file, output_file, mor_op, mode, wait_key_time)

if __name__ == "__main__":
    main(sys.argv[1:])