import cv2
import mediapipe as mp
import json
import time
import numpy as np
import os
from collections import deque
from math import sqrt

# --- PENGATURAN GLOBAL ---
CONFIG_PATH = "GestureMeme/gestures.json"
MEMES_FOLDER = "memes"
MEME_DISPLAY_DURATION = 3
GESTURE_THRESHOLD = 0.4  # 降低阈值以提高准确性
STABILITY_FRAMES = 5  # 需要连续检测到相同手势的帧数
MIN_GESTURE_INTERVAL = 1.0  # 手势之间的最小间隔时间(秒)
WINDOW_NAME = 'Auto Gesture Meme Cam'

# 人脸检测相关设置
FACE_DETECTION_CONFIDENCE = 0.5  # 人脸检测置信度阈值
FACE_REQUIRED_FOR_GESTURE = True  # 是否要求检测到人脸才进行手势识别

# --- Konstanta untuk Mode ---
MENU_MODE = "MENU"
ADD_MODE = "ADD"
LIVE_MODE = "LIVE"

# --- INISIALISASI MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 修改为检测2只手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# 添加人脸检测初始化
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=FACE_DETECTION_CONFIDENCE
)


# --- FUNGSI-FUNGSI PENDUKUNG ---

def extract_advanced_features(landmarks):
    """提取更高级的手势特征，包括手指角度和相对位置"""
    # 转换为numpy数组以便计算
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    # 计算手腕作为参考点
    wrist = points[0]

    # 计算每个手指的关键点
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]

    # 计算手指关节点
    index_pip = points[6]
    middle_pip = points[10]
    ring_pip = points[14]
    pinky_pip = points[18]

    # 计算手指之间的角度
    features = []

    # 1. 手指之间的角度
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        cos_angle = dot_product / (norm_v1 * norm_v2)
        # 确保值在有效范围内
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        return np.arccos(cos_angle)

    # 计算手指之间的角度
    index_to_middle = angle_between_vectors(index_tip - wrist, middle_tip - wrist)
    middle_to_ring = angle_between_vectors(middle_tip - wrist, ring_tip - wrist)
    ring_to_pinky = angle_between_vectors(ring_tip - wrist, pinky_tip - wrist)
    thumb_to_index = angle_between_vectors(thumb_tip - wrist, index_tip - wrist)

    features.extend([index_to_middle, middle_to_ring, ring_to_pinky, thumb_to_index])

    # 2. 手指弯曲度 (指尖到指关节的距离)
    index_bend = np.linalg.norm(index_tip - index_pip)
    middle_bend = np.linalg.norm(middle_tip - middle_pip)
    ring_bend = np.linalg.norm(ring_tip - ring_pip)
    pinky_bend = np.linalg.norm(pinky_tip - pinky_pip)

    features.extend([index_bend, middle_bend, ring_bend, pinky_bend])

    # 3. 手指之间的相对距离
    index_middle_dist = np.linalg.norm(index_tip - middle_tip)
    middle_ring_dist = np.linalg.norm(middle_tip - ring_tip)
    ring_pinky_dist = np.linalg.norm(ring_tip - pinky_tip)
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

    features.extend([index_middle_dist, middle_ring_dist, ring_pinky_dist, thumb_index_dist])

    # 4. 添加归一化的landmark作为补充特征
    normalized = normalize_landmarks(landmarks).flatten()
    features.extend(normalized[:20])  # 只取前20个点以减少特征维度

    return np.array(features)


def normalize_landmarks(landmarks):
    """Menormalkan posisi landmark agar posisi & ukuran tangan tidak berpengaruh."""
    wrist = landmarks.landmark[0]
    normalized_list = []
    for lm in landmarks.landmark:
        normalized_list.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return np.array(normalized_list)


def load_gestures():
    """Memuat data gestur dari file JSON."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except:
        return {}


def save_gestures(gestures_data):
    """Menyimpan data gestur ke file JSON."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(gestures_data, f, indent=4)


def weighted_distance(features1, features2, weights=None):
    """计算加权距离，给重要特征更高的权重"""
    if weights is None:
        # 默认权重：角度特征权重更高
        weights = np.ones(len(features1))
        # 前4个是角度特征，权重更高
        weights[:4] = 2.0
        # 接下来4个是弯曲度特征，权重中等
        weights[4:8] = 1.5
        # 最后是距离特征和归一化landmark
        weights[8:] = 1.0

    diff = features1 - features2
    weighted_diff = diff * weights
    return np.linalg.norm(weighted_diff)


def find_closest_gesture(current_features, stored_gestures):
    """Mencari gestur yang paling mirip dengan gestur saat ini."""
    if not stored_gestures:
        return None, float('inf')

    min_distance = float('inf')
    closest_gesture_name = None
    for name, data in stored_gestures.items():
        stored_features = np.array(data['features'])
        distance = weighted_distance(current_features, stored_features)
        if distance < min_distance:
            min_distance = distance
            closest_gesture_name = name

    return closest_gesture_name, min_distance


def draw_face_detections(frame, face_detections):
    """在帧上绘制人脸检测结果"""
    if face_detections:
        for detection in face_detections:  # 修复：直接遍历face_detections列表
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            # 绘制人脸边界框
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # 添加人脸标签
            cv2.putText(frame, "Wajah Terdeteksi",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


# --- FUNGSI PEMILIH KAMERA ---

def list_available_cameras():
    """Mencoba menemukan dan mencantumkan indeks kamera yang tersedia."""
    print("\nMencari kamera yang tersedia...")
    available_indices = []
    # Coba hingga 10 indeks, seharusnya cukup untuk sebagian besar sistem
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Gunakan CAP_DSHOW untuk Windows
        if cap.isOpened():
            # Coba dapatkan nama backend untuk info tambahan
            backend_name = cap.getBackendName()
            print(f"  Kamera ditemukan di indeks {i} (Backend: {backend_name})")
            available_indices.append(i)
            cap.release()  # Lepaskan segera agar tidak terkunci
    return available_indices


def select_camera():
    """Menampilkan daftar kamera dan meminta pengguna untuk memilih."""
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("❌ Tidak ada kamera yang ditemukan. Pastikan kamera terhubung dan coba lagi.")
        return None

    while True:
        try:
            choice_str = input(f"\nMasukkan indeks kamera yang ingin digunakan (contoh: {available_cameras[0]}): ")
            choice = int(choice_str)
            if choice in available_cameras:
                print(f"✅ Kamera di indeks {choice} dipilih.")
                return choice
            else:
                print(f"❌ Indeks {choice} tidak valid. Silakan pilih dari daftar di atas.")
        except ValueError:
            print("❌ Input tidak valid. Masukkan angka.")


# --- FUNGSI-FUNGSI UNTUK SETIAP MODE ---

def show_menu():
    """Menampilkan menu utama dan menunggu input dari user."""
    while True:
        # Buat gambar menu
        menu_img = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w, _ = menu_img.shape

        # Judul
        cv2.putText(menu_img, "GESTURE MEME CAM", (w // 2 - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(menu_img, "Dukungan 2 Tangan + Deteksi Wajah", (w // 2 - 160, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(menu_img, "Pilih Mode:", (w // 2 - 90, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Definisikan tombol
        button_add = {'text': '1. Tambah Gestur', 'pos': (w // 2 - 120, 240),
                      'rect': ((w // 2 - 150, 210), (w // 2 + 150, 270))}
        button_live = {'text': '2. Live Kamera', 'pos': (w // 2 - 110, 340),
                       'rect': ((w // 2 - 150, 310), (w // 2 + 150, 370))}
        button_exit = {'text': '3. Keluar', 'pos': (w // 2 - 70, 440),
                       'rect': ((w // 2 - 150, 410), (w // 2 + 150, 470))}

        # Gambar tombol
        for btn in [button_add, button_live, button_exit]:
            cv2.rectangle(menu_img, btn['rect'][0], btn['rect'][1], (0, 255, 0), 2)
            cv2.putText(menu_img, btn['text'], btn['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, menu_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            return ADD_MODE
        elif key == ord('2'):
            return LIVE_MODE
        elif key == ord('3') or key == 27:  # 27 adalah tombol ESC
            return "EXIT"


def add_gesture_mode(camera_index):
    """Mode untuk merekam gestur baru dengan tampilan visual yang lebih baik."""
    print("\n--- MODE TAMBAH GESTUR BARU ---")
    print("Catatan: Sistem akan mendeteksi hingga 2 tangan dan wajah")

    # Input dari user di console
    gesture_name = input("Masukkan nama untuk gestur baru (contoh: ok_sign): ").strip().lower()
    if not gesture_name:
        print("Nama gestur tidak boleh kosong. Kembali ke menu.")
        return

    gesture_desc = input(f"Masukkan deskripsi untuk '{gesture_name}' (contoh: OK Sign 👌): ").strip()

    # Tampilkan daftar file di folder memes
    print(f"\nFile gambar yang tersedia di folder '{MEMES_FOLDER}/':")
    if not os.path.exists(MEMES_FOLDER):
        os.makedirs(MEMES_FOLDER)

    meme_files = [f for f in os.listdir(MEMES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if not meme_files:
        print("  (Tidak ada gambar. Silakan tambahkan gambar ke folder 'memes' terlebih dahulu.)")
        return

    for i, f in enumerate(meme_files):
        print(f"  {i + 1}. {f}")

    try:
        choice = int(input("Pilih nomor gambar: ")) - 1
        if 0 <= choice < len(meme_files):
            image_filename = meme_files[choice]
        else:
            print("Pilihan tidak valid. Kembali ke menu.")
            return
    except ValueError:
        print("Input tidak valid. Kembali ke menu.")
        return

    image_path = os.path.join(MEMES_FOLDER, image_filename)

    # Proses perekaman dengan tampilan visual
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Tidak bisa mengakses kamera di indeks {camera_index}.")
        return

    # Tampilan countdown
    print("\nSiapkan gestur Anda. Perekaman akan dimulai dalam...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        # Tampilkan countdown di kamera
        start_time = time.time()
        while time.time() - start_time < 1:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)

            # 添加人脸检测
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb_frame)
            # 修复：检查face_results.detections是否存在
            if face_results.detections:
                frame = draw_face_detections(frame, face_results.detections)

            # Tambahkan teks countdown
            cv2.putText(frame, str(i), (frame.shape[1] // 2 - 50, frame.shape[0] // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            cv2.putText(frame, f"Persiapkan gestur '{gesture_name}'", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

    print("REKAM! Jaga gestur Anda...")

    samples = []
    sample_count = 0
    target_samples = 30  # Jumlah sample yang akan diambil
    hand_detected = False
    face_detected = False

    # Tambahkan progress bar
    progress_bar_width = 400
    progress_bar_height = 30
    progress_bar_x = (640 - progress_bar_width) // 2
    progress_bar_y = 420

    while sample_count < target_samples:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 添加人脸检测
        face_results = face_detection.process(rgb_frame)
        # 修复：检查face_results.detections是否存在
        face_detected = face_results.detections is not None and len(face_results.detections) > 0

        # Reset frame dengan overlay informasi
        info_frame = frame.copy()

        # 添加人脸检测结果到画面
        if face_results.detections:
            info_frame = draw_face_detections(info_frame, face_results.detections)

        # Tambahkan judul
        cv2.putText(info_frame, f"Merekam gestur: {gesture_name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Tambahkan instruksi
        cv2.putText(info_frame, "Tunjukkan gestur Anda ke kamera", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加人脸检测状态
        face_status = "Wajah Terdeteksi" if face_detected else "Wajah Tidak Terdeteksi"
        face_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(info_frame, f"Status: {face_status}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

        # Progress bar background
        cv2.rectangle(info_frame, (progress_bar_x, progress_bar_y),
                      (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
                      (50, 50, 50), -1)

        # Progress bar fill
        progress_width = int((sample_count / target_samples) * progress_bar_width)
        cv2.rectangle(info_frame, (progress_bar_x, progress_bar_y),
                      (progress_bar_x + progress_width, progress_bar_y + progress_bar_height),
                      (0, 255, 0), -1)

        # Progress text
        progress_text = f"{sample_count}/{target_samples}"
        cv2.putText(info_frame, progress_text,
                    (progress_bar_x + progress_bar_width // 2 - 30, progress_bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            hand_detected = True

            # 处理多只手
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 为不同手使用不同颜色
                color = (0, 255, 0) if idx == 0 else (255, 0, 0)  # 第一只手绿色，第二只手蓝色

                # Gambar hand landmarks dengan style yang lebih menarik
                mp_draw.draw_landmarks(
                    info_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )

                # 添加手部标签
                handedness = results.multi_handedness[idx].classification[0].label
                label = f"Tangan {idx + 1} ({handedness})"
                cv2.putText(info_frame, label, (50, 150 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 检查是否需要人脸检测
            can_record = not FACE_REQUIRED_FOR_GESTURE or face_detected

            if can_record:
                # 只使用第一只手进行特征提取（可以根据需要修改）
                if results.multi_hand_landmarks[0]:
                    # Ambil sample setiap beberapa frame
                    if sample_count % 3 == 0:  # Ambil sample setiap 3 frame
                        # 使用新的特征提取方法
                        features = extract_advanced_features(results.multi_hand_landmarks[0])
                        samples.append(features)

                    sample_count += 1
            else:
                # 添加提示需要人脸检测
                cv2.putText(info_frame, "Perlu deteksi wajah untuk merekam gestur", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Tambahkan indikator visual ketika tangan tidak terdeteksi
            cv2.circle(info_frame, (50, 150), 20, (0, 0, 255), -1)
            cv2.putText(info_frame, "Tidak Ada Tangan", (80, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Tambahkan pesan peringatan
            cv2.putText(info_frame, "Harap tunjukkan tangan Anda ke kamera", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, info_frame)
        cv2.waitKey(1)

    # Tampilkan hasil perekaman
    if samples:
        avg_features = np.mean(samples, axis=0).tolist()
        gestures_data = load_gestures()
        gestures_data[gesture_name] = {
            "description": gesture_desc,
            "image": image_path,
            "features": avg_features  # 存储新的特征而不是landmarks
        }
        save_gestures(gestures_data)

        # Tampilkan pesan sukses di kamera
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            # 添加人脸检测
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb_frame)
            # 修复：检查face_results.detections是否存在
            if face_results.detections:
                frame = draw_face_detections(frame, face_results.detections)

            # Tambahkan overlay sukses
            overlay = frame.copy()
            cv2.rectangle(overlay, (50, 50), (590, 430), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.putText(frame, "GESTUR BERHASIL DIREKAM!", (120, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Nama: {gesture_name}", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Deskripsi: {gesture_desc}", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Tekan tombol apa saja untuk melanjutkan", (120, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(0)

        print(f"\n✅ Gestur '{gesture_name}' berhasil direkam dan disimpan!")
    else:
        # Tampilkan pesan gagal di kamera
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            # 添加人脸检测
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb_frame)
            # 修复：检查face_results.detections是否存在
            if face_results.detections:
                frame = draw_face_detections(frame, face_results.detections)

            # Tambahkan overlay gagal
            overlay = frame.copy()
            cv2.rectangle(overlay, (50, 50), (590, 430), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.putText(frame, "PEKERAMAN GAGAL!", (180, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, "Tidak ada tangan yang terdeteksi", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "selama proses perekaman", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Tekan tombol apa saja untuk melanjutkan", (120, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(0)

        print("❌ Tidak ada tangan yang terdeteksi selama perekaman.")

    cap.release()
    cv2.destroyAllWindows()
    print("--- KEMBALI KE MENU UTAMA ---\n")
    time.sleep(2)


def live_camera_mode(camera_index):
    """Mode kamera live untuk mendeteksi gestur."""
    print("\n--- MASUK MODE LIVE KAMERA ---")
    print("Tekan 'm' untuk kembali ke menu utama.")
    print("Mendeteksi hingga 2 tangan dan wajah...")

    gestures_data = load_gestures()
    if not gestures_data:
        print("Belum ada gestur yang tersimpan. Tambahkan gestur terlebih dahulu.")
        time.sleep(2)
        return

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Tidak bisa mengakses kamera di indeks {camera_index}.")
        return

    show_meme = False
    meme_start_time = 0
    current_meme_image = None

    # 添加手势稳定性检查 - 为每只手单独跟踪
    gesture_history = [deque(maxlen=STABILITY_FRAMES), deque(maxlen=STABILITY_FRAMES)]
    last_gesture_time = [0, 0]
    last_gesture_name = [None, None]

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        if show_meme:
            if time.time() - meme_start_time > MEME_DISPLAY_DURATION:
                show_meme = False
                current_meme_image = None
            else:
                if current_meme_image is not None:
                    meme_resized = cv2.resize(current_meme_image, (frame.shape[1], frame.shape[0]))
                    cv2.imshow(WINDOW_NAME, meme_resized)
                key = cv2.waitKey(5)
                if key & 0xFF == ord('m'):
                    break
                continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 添加人脸检测
        face_results = face_detection.process(rgb_frame)
        # 修复：检查face_results.detections是否存在
        face_detected = face_results.detections is not None and len(face_results.detections) > 0

        # 在画面上绘制人脸检测结果
        if face_results.detections:
            frame = draw_face_detections(frame, face_results.detections)

        triggered_gesture = None
        confidence = 0

        # 添加人脸检测状态指示器
        face_status = "Wajah Terdeteksi" if face_detected else "Wajah Tidak Terdeteksi"
        face_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, face_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

        if results.multi_hand_landmarks:
            # 处理检测到的每只手
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 为不同手使用不同颜色
                color = (0, 255, 0) if idx == 0 else (255, 0, 0)  # 第一只手绿色，第二只手蓝色

                # Gambar hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )

                # 获取手部信息
                handedness = results.multi_handedness[idx].classification[0].label
                label = f"Tangan {idx + 1} ({handedness})"

                # 使用新的特征提取方法
                current_features = extract_advanced_features(hand_landmarks)
                gesture_name, distance = find_closest_gesture(current_features, gestures_data)

                # 计算置信度
                confidence = max(0, 1 - (distance / GESTURE_THRESHOLD))

                # 添加到对应手的历史记录
                if distance < GESTURE_THRESHOLD:
                    gesture_history[idx].append(gesture_name)
                else:
                    gesture_history[idx].append(None)

                # 检查手势是否稳定
                if len(gesture_history[idx]) == STABILITY_FRAMES:
                    # 检查是否所有最近的帧都检测到相同的手势
                    if all(g == gesture_history[idx][0] for g in gesture_history[idx]) and gesture_history[idx][
                        0] is not None:
                        current_time = time.time()
                        # 检查是否超过最小间隔时间
                        if (current_time - last_gesture_time[idx] > MIN_GESTURE_INTERVAL or
                                gesture_history[idx][0] != last_gesture_name[idx]):
                            # 检查是否需要人脸检测
                            if not FACE_REQUIRED_FOR_GESTURE or face_detected:
                                triggered_gesture = gesture_history[idx][0]
                                last_gesture_time[idx] = current_time
                                last_gesture_name[idx] = gesture_history[idx][0]

                # 显示当前检测到的手势和置信度
                text_y = 90 + idx * 30
                if gesture_name:
                    cv2.putText(frame, f"{label}: {gesture_name} ({confidence:.2f})",
                                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, f"{label}: Tidak terdeteksi",
                                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if triggered_gesture:
            config = gestures_data[triggered_gesture]
            image_path = config["image"]
            current_meme_image = cv2.imread(image_path)
            if current_meme_image is not None:
                show_meme = True
                meme_start_time = time.time()
                print(f"✅ {config['description']} terdeteksi!")
            else:
                print(f"⚠️ Gagal memuat gambar: {image_path}")

        cv2.putText(frame, "Mode: LIVE (2 Tangan + Deteksi Wajah)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.putText(frame, "Tekan 'm' untuk menu", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('m'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- KEMBALI KE MENU UTAMA ---\n")


# --- PROGRAM UTAMA ---

def main():
    """Fungsi utama untuk menjalankan aplikasi."""
    # Pilih kamera di awal
    selected_camera_index = select_camera()
    if selected_camera_index is None:
        print("Program ditutup karena tidak ada kamera yang dipilih.")
        return  # Keluar jika tidak ada kamera

    current_mode = MENU_MODE

    while True:
        if current_mode == MENU_MODE:
            next_mode = show_menu()
            if next_mode == "EXIT":
                break
            current_mode = next_mode

        elif current_mode == ADD_MODE:
            # Kirim indeks kamera yang dipilih
            add_gesture_mode(selected_camera_index)
            current_mode = MENU_MODE

        elif current_mode == LIVE_MODE:
            # Kirim indeks kamera yang dipilih
            live_camera_mode(selected_camera_index)
            current_mode = MENU_MODE

    print("Program ditutup. Sampai jumpa!")
    cv2.destroyAllWindows()
    hands.close()
    face_detection.close()


if __name__ == '__main__':
    main()
