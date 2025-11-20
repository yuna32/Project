import numpy as np
import pandas as pd
import glob
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- 머신러닝 라이브러리 ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# (앱에서 정의한 10개 단어 리스트 - 시각화용)
WORD_CLASSES = [
    "Security", "Lock", "Password", "Access", "Allow",
    "Agree", "Confidential", "Protect", "Authority", "Delete"
]

# --- 헬퍼 함수 정의 ---

def lowpass_filter(x, fs=100, cutoff=10, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, x, axis=0)

def extract_features(time_series_sample):
    f_mean = np.mean(time_series_sample, axis=0)
    f_std = np.std(time_series_sample, axis=0)
    f_min = np.min(time_series_sample, axis=0)
    f_max = np.max(time_series_sample, axis=0)
    f_median = np.median(time_series_sample, axis=0)
    return np.concatenate([f_mean, f_std, f_min, f_max, f_median], axis=0)

def load_and_preprocess_resampled(csv_file, target_fs=100.0):

    try:
        # 1. CSV 로딩
        df = pd.read_csv(csv_file)
        
        # 2. 데이터 클리닝 ('########' 제거)
        df.replace('########', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # 데이터가 너무 짧으면 스킵 
        if len(df) < 10: 
            print(f"Skipping too short file: {csv_file}")
            return None, None

        # 3. 타입 변환 및 정렬
        df = df.astype({'timestamp': float, 'ax': float, 'ay': float, 'az': float, 
                        'gx': float, 'gy': float, 'gz': float, 'label': int})
        df.sort_values(by='timestamp', inplace=True)

        # 4. 리샘플링 (100Hz로 균일하게 맞춤)
        cols_to_process = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        t_original = df['timestamp'].values
        
        # 타임스탬프 중복 제거 (드물게 발생 가능)
        unique_indices = np.unique(t_original, return_index=True)[1]
        t_original = t_original[unique_indices]
        data_original = df[cols_to_process].values[unique_indices]

        if len(t_original) < 2: return None, None

        t_start = t_original[0]
        t_end = t_original[-1]
        # 0.01초(100Hz) 간격의 새로운 타임스탬프 생성
        t_target = np.arange(t_start, t_end, 1.0 / target_fs)
        
        if len(t_target) < 2: return None, None

        resampled_data = np.zeros((len(t_target), len(cols_to_process)))
        for i in range(len(cols_to_process)):
            sig_original = data_original[:, i]
            f_interp = interp1d(t_original, sig_original, kind='linear', 
                                bounds_error=False, fill_value="extrapolate")
            resampled_data[:, i] = f_interp(t_target)

        # 5. 필터링
        filtered_data = lowpass_filter(resampled_data, fs=target_fs, cutoff=10)
        
        # 레이블 추출 (0~9 사이의 단어 인덱스)
        label = df['label'].iloc[0]
        return filtered_data, int(label)

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None, None

# --- 메인 실행 코드 ---

TARGET_FS = 100.0
data_list = []
labels_list = []

print("Starting preprocessing (Loading CSVs ...")

csv_files = glob.glob("data/*.csv")
print(f"Found {len(csv_files)} CSV files.")

for f in csv_files:
    resampled_sample, label = load_and_preprocess_resampled(f, target_fs=TARGET_FS)
    
    if resampled_sample is not None:
        data_list.append(resampled_sample)
        labels_list.append(label)

print(f"Successfully loaded {len(data_list)} samples.")

if not data_list:
    print("No data loaded. Check your 'data' folder path.")
    exit()

# --- 데이터 불균형 확인 및 필터링 ---
print("\nChecking class distribution...")
label_counts = Counter(labels_list)
print(label_counts) # 예: {0: 30, 1: 29, ...}

# 최소 2개 이상 샘플이 있는 클래스만 사용 (Stratified Split을 위해)
labels_to_remove = {label for label, count in label_counts.items() if count < 2}
if labels_to_remove:
    print(f"WARNING: Removing classes with < 2 samples: {labels_to_remove}")
    data_list, labels_list = zip(*[(d, l) for d, l in zip(data_list, labels_list) if l not in labels_to_remove])
    data_list = list(data_list)
    labels_list = list(labels_list)

# --- 훈련 / 테스트 분리 ---
X_train, X_test, y_train, y_test = train_test_split(
    data_list, labels_list, test_size=0.2, random_state=42, stratify=labels_list
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# --- 표준화 ---
# 훈련 데이터 전체를 합쳐서 스케일러 학습
all_train_data = np.concatenate(X_train, axis=0)
scaler = StandardScaler()
scaler.fit(all_train_data)

# 각 샘플을 스케일링 
X_train_scaled = [scaler.transform(sample) for sample in X_train]
X_test_scaled = [scaler.transform(sample) for sample in X_test]
print("Standardization complete.")

# --- 피처 추출 (Feature Extraction) ---
print("\nExtracting features...")
# (N_samples, 30) 형태의 2D 배열로 변환
X_train_features = np.array([extract_features(sample) for sample in X_train_scaled])
X_test_features = np.array([extract_features(sample) for sample in X_test_scaled])
y_train_array = np.array(y_train)
y_test_array = np.array(y_test)

print(f"Feature matrix shape: {X_train_features.shape}")

# --- RandomForest 모델 훈련 ---
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200, 
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_features, y_train_array)
print("Model trained.")

# --- 평가 ---
print("\nPredicting on test set...")
y_pred = rf_model.predict(X_test_features)
accuracy = accuracy_score(y_test_array, y_pred)

print(f"\n=== Command Classification Results ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_array, y_pred))

# --- Confusion Matrix 시각화 ---
print("Displaying Confusion Matrix...")
cm = confusion_matrix(y_test_array, y_pred)
unique_labels = sorted(np.unique(y_test_array))
# 그래프에 0,1,2 대신 실제 단어(보안, 잠금...)를 표시하기 위해 매핑
tick_labels = [WORD_CLASSES[i] for i in unique_labels if i < len(WORD_CLASSES)]

plt.figure(figsize=(10, 8)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=tick_labels, yticklabels=tick_labels)
plt.title('Command Word Classification Confusion Matrix')
plt.xlabel('Predicted Command')
plt.ylabel('True Command')
plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()
