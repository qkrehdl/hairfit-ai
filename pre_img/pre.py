import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# --- 1. MediaPipe 설정 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# --- 2. 분석에 필요한 랜드마크 인덱스 정의 ---
# 얼굴 길이 (L): 미간 중앙(10) - 턱 끝(152)
# 얼굴 폭 (W): 광대뼈 좌(35) - 광대뼈 우(265)
# 턱 폭 (J): 하관 좌(58) - 하관 우(288)
LANDMARK_INDICES = {
    'forehead_center': 10,
    'chin_tip': 152,
    'cheek_left': 35,
    'cheek_right': 265,
    'jaw_left': 58,
    'jaw_right': 288
}

# --- 3. 유니코드(한글 경로) 안전 이미지 로드 함수 ---
def imread_unicode(path):
    """
    Windows + 한글 경로에서도 안전하게 이미지 로드
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        if stream.size == 0:
            return None
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [치명적 오류] imread_unicode 실패: {path}, 에러: {e}")
        return None

# --- 4. 유클리드 거리 계산 ---
def calculate_distance(p1, p2):
    """
    MediaPipe landmark (정규화 좌표) 기준 유클리드 거리
    → 절대 길이가 아닌 '비율' 비교용
    """
    return np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

# --- 5. 데이터 분석 함수 ---
def analyze_images(base_dir='photos'):
    results = []

    for root, dirs, files in os.walk(base_dir):
        face_type = os.path.basename(root)

        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                continue

            filepath = os.path.join(root, filename)

            # 기본 데이터 행
            data_row = {
                'filename': filename,
                'face_type': face_type,
                'face_detected': False,
                'L_W_Ratio': np.nan,
                'J_W_Ratio': np.nan
            }

            # --- 이미지 로드 (한글 경로 대응) ---
            image = imread_unicode(filepath)

            if image is None:
                print(f"  [오류] 이미지 로드 실패: {filepath}")
                results.append(data_row)
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # --- MediaPipe Face Mesh 실행 ---
            mp_result = face_mesh.process(image_rgb)

            if not mp_result.multi_face_landmarks:
                results.append(data_row)
                continue

            data_row['face_detected'] = True

            try:
                landmarks = mp_result.multi_face_landmarks[0].landmark
                p = {k: landmarks[v] for k, v in LANDMARK_INDICES.items()}

                L = calculate_distance(p['forehead_center'], p['chin_tip'])
                W = calculate_distance(p['cheek_left'], p['cheek_right'])
                J = calculate_distance(p['jaw_left'], p['jaw_right'])

                if W > 0:
                    data_row['L_W_Ratio'] = L / W
                    data_row['J_W_Ratio'] = J / W

            except Exception as e:
                print(f"  [경고] {filename} 랜드마크 계산 실패: {e}")

            results.append(data_row)

    return pd.DataFrame(results)

# --- 6. 코드 실행 ---
if __name__ == "__main__":

    if not os.path.exists('photos'):
        os.makedirs('photos')
        print("'photos' 폴더가 생성되었습니다.")
        print("안에 얼굴형별 폴더(긴형, 둥근형, 계란형, 각진형)를 만들고 이미지를 넣어주세요.")
    else:
        df_analysis = analyze_images(base_dir='photos')

        output_filename = 'face_shape_analysis_results.csv'
        df_analysis.to_csv(
            output_filename,
            index=False,
            encoding='utf-8-sig'   # 엑셀 한글 깨짐 방지
        )

        print("\n====================================")
        print(f"✅ 분석 완료: {output_filename}")
        print(f"총 이미지 수: {len(df_analysis)}")
        print(f"얼굴 감지 성공: {df_analysis['face_detected'].sum()} / {len(df_analysis)}")
        print("CSV에서 L/W, J/W 평균·분포를 확인해 임계값을 설정하세요.")
        print("====================================")
