import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Hàm tiền xử lý ảnh
def preprocess_image(image_path, size=(256, 256)):
    # Đọc ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn: ".format(image_path))
    # Resize ảnh
    img = cv2.resize(img, size)
    # Chuẩn hóa giá trị pixel về [0, 1]
    img = img.astype(np.float32) / 255.0
    return img

# Hàm áp dụng FFT và lọc nhiễu, làm nổi bật khối u với dịch chuyển ROI
def apply_fft_and_filter(image, r=30, roi_size=80, shift_y=-20):
    # Áp dụng FFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Tạo phổ tần số
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-10)  # Tránh log(0)
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))

    # Tạo bộ lọc thông thấp (low-pass filter) để loại bỏ tần số cao
    rows, cols = image.shape
    crow, ccol = rows // 2 + shift_y, cols // 2  # Dịch trung tâm lên trên với shift_y
    mask = np.ones((rows, cols, 2), dtype=np.uint8)  # Khởi tạo mask với giá trị 1
    r = r  # Bán kính bộ lọc
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    # Loại bỏ tần số cao ngoài bán kính r
    mask_area_high = (x - center[0]) ** 2 + (y - center[1]) ** 2 > r * r
    mask[mask_area_high] = 0

    # Tạo mask ROI để thu hẹp vùng quan tâm quanh khối u
    mask_roi = np.zeros((rows, cols), dtype=np.uint8)
    roi_half = roi_size // 2
    # Đảm bảo ROI không vượt ra ngoài biên ảnh
    start_row = max(0, crow - roi_half)
    end_row = min(rows, crow + roi_half)
    start_col = max(0, ccol - roi_half)
    end_col = min(cols, ccol + roi_half)
    mask_roi[start_row:end_row, start_col:end_col] = 1
    mask_roi = np.repeat(mask_roi[:, :, np.newaxis], 2, axis=2)  # Mở rộng mask cho 2 kênh

    # Kết hợp mask tần số cao và mask ROI
    final_mask = mask * mask_roi
    fshift = dft_shift * final_mask
    f_ishift = np.fft.ifftshift(fshift)

    # Biến đổi Fourier ngược (inverted FFT)
    img_back = cv2.idft(f_ishift)
    filtered_img = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Chuẩn hóa và tăng độ tương phản cho vùng ROI
    filtered_img = (filtered_img - np.min(filtered_img)) / (np.max(filtered_img) - np.min(filtered_img))
    # Giữ lại chỉ vùng ROI, đặt phần còn lại về 0
    filtered_img = filtered_img * mask_roi[:, :, 0]
    # Chuyển về uint8 và tăng độ tương phản
    filtered_img_uint8 = np.clip(filtered_img * 255, 0, 255).astype(np.uint8)
    filtered_img_uint8 = cv2.equalizeHist(filtered_img_uint8)
    # Áp dụng CLAHE để làm nổi bật khối u
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    filtered_img_uint8 = clahe.apply(filtered_img_uint8)
    filtered_img = filtered_img_uint8.astype(np.float32) / 255.0

    return magnitude_spectrum, filtered_img

# Hàm trích xuất đặc trưng tần số
def extract_frequency_features(magnitude_spectrum):
    mean_magnitude = np.mean(magnitude_spectrum)
    std_magnitude = np.std(magnitude_spectrum)
    return [mean_magnitude, std_magnitude]

# Hàm phân loại khối u (ví dụ với SVM)
def classify_tumor(features, labels, test_features):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    test_features = scaler.transform([test_features])

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(features, labels)
    prediction = clf.predict(test_features)
    return prediction

# Chủ đạo chương trình
def main(image_path, r=30, roi_size=80, shift_y=-20):
    # Bước 1: Tiền xử lý
    img = preprocess_image(image_path)

    # Bước 2: Áp dụng FFT và lọc nhiễu, làm nổi bật khối u với dịch chuyển ROI
    magnitude_spectrum, filtered_img = apply_fft_and_filter(img, r=r, roi_size=roi_size, shift_y=shift_y)

    # Bước 3: Trích xuất đặc trưng
    features = extract_frequency_features(magnitude_spectrum)

    # Bước 4: Phân loại (giả định có dữ liệu huấn luyện)
    # dummy_features = [[100, 20], [150, 25], [80, 15]]  # Ví dụ đặc trưng huấn luyện
    # dummy_labels = [0, 1, 0]  # 0: lành tính, 1: ác tính
    # prediction = classify_tumor(dummy_features, dummy_labels, features)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Ảnh gốc')
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Phổ tần số')
    plt.subplot(133), plt.imshow(filtered_img, cmap='gray'), plt.title('Ảnh với khối u nổi bật')
    plt.show()

    # print(f"Kết quả phân loại: {'Ác tính' if prediction[0] == 1 else 'Lành tính'}")
    # print(f"Đặc trưng tần số: {features}")

if __name__ == "__main__":
    # Thay bằng đường dẫn đến ảnh MRI của bạn
    input_image_path = "mri_image.jpg"
    main(input_image_path, r=45, roi_size=80, shift_y=-35)  # shift_y âm để dịch lên trên