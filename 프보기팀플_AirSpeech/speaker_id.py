import numpy as np
import pandas as pd
import glob
import os 
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


# --- 오디오 및 머신러닝 라이브러리 추가 ---
import librosa 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 헬퍼 함수 정의 (로우패스, 피처 추출) ---

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

# --- VAD 로직이 포함된 데이터 로더 ---
def load_and_preprocess_resampled(csv_file, wav_file, target_fs=100.0, vad_top_db=30):
    try:
        # 1. 오디오(WAV)로 VAD 수행 -> 시작/종료 시간 획득
        audio, sr = librosa.load(wav_file, sr=None)
        # top_db=30: 조용한 부분(평균-30dB)을 침묵으로 간주하고 잘라냄 (환경에 따라 조절)
        audio_trimmed, index = librosa.effects.trim(audio, top_db=vad_top_db)
        
        if len(index) == 0: # 오디오가 비어있으면 스킵
            print(f"Skipping empty audio: {wav_file}")
            return None, None
            
        start_time = index[0] / sr # 발화 시작 시간 (초)
        end_time = index[1] / sr   # 발화 종료 시간 (초)

        # 2. IMU(CSV) 데이터 로딩 및 클리닝
        df = pd.read_csv(csv_file)
        df.replace('########', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # 3. VAD 시간으로 IMU 데이터 필터링
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        # 4. 리샘플링을 위한 전처리
        if len(df) < 20: 
            print(f"Skipping short file (after VAD): {csv_file}")
            return None, None
            
        df.sort_values(by='timestamp', inplace=True)
        df = df.astype({'timestamp': float, 'ax': float, 'ay': float, 'az': float, 
                        'gx': float, 'gy': float, 'gz': float, 'label': int})
        
        cols_to_process = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        t_original = df['timestamp'].values
        unique_indices = np.unique(t_original, return_index=True)[1]
        t_original = t_original[unique_indices]
        data_original = df[cols_to_process].values[unique_indices]
        
        if len(t_original) < 2:
            return None, None
            
        t_start = t_original[0]
        t_end = t_original[-1]
        t_target = np.arange(t_start, t_end, 1.0 / target_fs)
        
        if len(t_target) == 0: # VAD 결과가 너무 짧아 타겟이 없는 경우
            return None, None
            
        resampled_data = np.zeros((len(t_target), len(cols_to_process)))
        for i in range(len(cols_to_process)):
            sig_original = data_original[:, i]
            f_interp = interp1d(t_original, sig_original, kind='linear', 
                                bounds_error=False, fill_value="extrapolate")
            resampled_data[:, i] = f_interp(t_target)
            
        filtered_data = lowpass_filter(resampled_data, fs=target_fs, cutoff=10)
        
       
        label = df['label'].iloc[0] 
        return filtered_data, int(label)
        
    except Exception as e:
        print(f"Error processing {csv_file} or {wav_file}: {e}")
        return None, None

# --- [1. 전처리] ---
TARGET_FS = 100.0
data_list = []
labels_list = []

print("Starting preprocessing with VAD...")
for wav_f in glob.glob("data/*.wav"):
    # 짝이 되는 .csv 파일 이름 생성
    csv_f = wav_f.replace(".wav", ".csv")
    
    if not os.path.exists(csv_f):
        print(f"Skipping {wav_f}, missing CSV file pair.")
        continue
    
    resampled_sample, label = load_and_preprocess_resampled(csv_f, wav_f, target_fs=TARGET_FS)
    
    if resampled_sample is not None:
        data_list.append(resampled_sample)
        labels_list.append(label)

print(f"Successfully loaded and processed {len(data_list)} samples.")

if not data_list:
    print("No data loaded. Exiting.")
    exit()

# --- [2. 레이블 < 2개 필터링] ---
print("\nChecking speaker label distribution before filtering...")
label_counts = Counter(labels_list)
print(label_counts) # 예: Counter({1: 20, 2: 19, 3: 21})
labels_to_remove = {label for label, count in label_counts.items() if count < 2}

if labels_to_remove:
    print(f"WARNING: Removing speakers with < 2 samples: {labels_to_remove}")
    filtered_data_list = []
    filtered_labels_list = []
    for sample, label in zip(data_list, labels_list):
        if label not in labels_to_remove:
            filtered_data_list.append(sample)
            filtered_labels_list.append(label)
    data_list = filtered_data_list
    labels_list = filtered_labels_list
    print(f"Data filtered. New total samples: {len(data_list)}")
    if not data_list:
        print("No data remaining after filtering. Exiting.")
        exit()
else:
    print("All speaker labels have sufficient samples (>= 2).")

# --- [3. 훈련/테스트 분리] ---
X_train, X_test, y_train, y_test = train_test_split(
    data_list, labels_list, test_size=0.2, random_state=42, stratify=labels_list
)

if not X_train or not X_test:
    print("Train or Test set is empty after split. Need more data.")
    exit()

print(f"\nTotal samples: {len(data_list)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# --- [4. 표준화] ---
all_train_data = np.concatenate(X_train, axis=0) 
scaler = StandardScaler()
scaler.fit(all_train_data) 
X_train_scaled = [scaler.transform(sample) for sample in X_train]
X_test_scaled = [scaler.transform(sample) for sample in X_test]
print("\nStandardization complete.")

# --- [5. 피처 추출] ---
print("\nExtracting features from scaled data...")
X_train_features = np.array([extract_features(sample) for sample in X_train_scaled])
X_test_features = np.array([extract_features(sample) for sample in X_test_scaled])
y_train_array = np.array(y_train)
y_test_array = np.array(y_test)
print(f"Feature matrix shape: {X_train_features.shape}") 
print(f"Labels array shape: {y_train_array.shape}")

# --- [6. RandomForest 모델 훈련] ---
print("\nInitializing Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
print("Fitting the model...")
rf_model.fit(X_train_features, y_train_array)
print("Model fitting complete.")

# --- [7. 모델 평가] ---
print("\nPredicting on test set...")
y_pred = rf_model.predict(X_test_features)
accuracy = accuracy_score(y_test_array, y_pred)
print(f"\n--- Random Forest (Feature-based Speaker ID) Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_array, y_pred))

# Confusion Matrix 시각화
print("Displaying Confusion Matrix...")
cm = confusion_matrix(y_test_array, y_pred)
# 레이블이 0,1,2... 가 아닌 1,2,3... 일 수 있으므로 정렬
labels = sorted(np.unique(y_test_array)) 

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title('Random Forest (Speaker ID) Confusion Matrix')
plt.xlabel('Predicted Speaker ID')
plt.ylabel('True Speaker ID')
plt.show()

print("\nScript finished.")
