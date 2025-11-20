import numpy as np
import pandas as pd
import glob
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis, iqr, entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- 머신러닝 라이브러리 ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 단어 리스트 (시각화용)
WORD_CLASSES = [
     "보안", "잠금", "암호", "접속", "허용",
        "동의", "기밀", "보호", "권한", "삭제"
]

# --- 1. 전처리 함수  ---

def lowpass_filter(x, fs=100, cutoff=10, order=4):
    """ 10Hz 이상 고주파 노이즈 제거 """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, x, axis=0)

def combine_axes_paper_method(df):
    """
    논문의 Eq 3, 4, 5 적용:
    각 축을 양수로 변환 -> 정규화 -> 결합
    """
    # 1. 각 축 데이터 가져오기
    x = df['ax'].values
    y = df['ay'].values
    z = df['az'].values
    
    # 노이즈 평균 추정 (파일 전체 평균을 사용)
    # 논문: x_positive = x - mean(x_noise) + 1
    # 여기서는 mean(x)를 mean(x_noise)의 근사치로 사용 (중력/오프셋 제거 효과)
    x_pos = x - np.mean(x) + 1
    y_pos = y - np.mean(y) + 1
    z_pos = z - np.mean(z) + 1
    
    # 2. 최대값으로 정규화 (Eq 4)
    # x = x_pos / max(x_pos, y_pos, z_pos)
    max_val = np.max([np.max(np.abs(x_pos)), np.max(np.abs(y_pos)), np.max(np.abs(z_pos))])
    
    if max_val == 0: return np.zeros_like(x)
    
    x_norm = x_pos / max_val
    y_norm = y_pos / max_val
    z_norm = z_pos / max_val
    
    # 3. 결합 (Eq 5)
    # data_com = sqrt(x^2 + y^2 + z^2)
    combined = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2)
    
    return combined

# --- 2. 피처 추출 함수 ---

def calculate_cdp(signal, n_bins=10):
    """
    Intensity Rhythm: CDP (Cumulative Distribution of Points)
    에너지의 누적 분포를 n_bins 등분하여 추출
    """
    # 에너지 계산 (제곱)
    energy = signal ** 2
    # 누적 합 (Cumulative Sum)
    cumsum = np.cumsum(energy)
    
    if cumsum[-1] == 0:
        return np.zeros(n_bins)
        
    # 정규화 (0~1 사이로)
    cumsum_norm = cumsum / cumsum[-1]
    
    # 균등한 간격으로 샘플링 (Resampling to n_bins)
    x_old = np.linspace(0, 1, len(cumsum_norm))
    x_new = np.linspace(0, 1, n_bins)
    
    f_interp = interp1d(x_old, cumsum_norm, kind='linear')
    cdp_features = f_interp(x_new)
    
    return cdp_features

def extract_features_paper(combined_signal, fs=100):
    """
    논문에 언급된 Time-Frequency Stats + Intensity Rhythm (CDP)
    """
    # 1. Basic Stats
    f_min = np.min(combined_signal)
    f_max = np.max(combined_signal)
    f_mean = np.mean(combined_signal)
    f_std = np.std(combined_signal)
    f_var = np.var(combined_signal)
    f_median = np.median(combined_signal)
    f_range = f_max - f_min
    
    # CV (Coefficient of Variation)
    f_cv = f_std / (f_mean + 1e-9)
    
    # Skewness, Kurtosis
    f_skew = skew(combined_signal)
    f_kurt = kurtosis(combined_signal)
    
    # Quartiles & IQR
    f_q1 = np.percentile(combined_signal, 25)
    f_q2 = np.percentile(combined_signal, 50)
    f_q3 = np.percentile(combined_signal, 75)
    f_iqr = f_q3 - f_q1
    
    # Mean Crossing Rate (MCR) - 평균을 교차하는 횟수
    zero_centered = combined_signal - f_mean
    f_mcr = ((zero_centered[:-1] * zero_centered[1:]) < 0).sum() / len(combined_signal)
    
    # Absolute Area (Sum)
    f_abs_area = np.sum(np.abs(combined_signal))
    
    # Power Spectral Entropy (주파수 영역 엔트로피)
    # 100Hz 샘플링이므로 50Hz까지의 스펙트럼
    try:
        f, Pxx = periodogram(combined_signal, fs) # Pxx is power spectral density
        f_entropy = entropy(Pxx)
    except:
        f_entropy = 0 # 길이가 너무 짧거나 오류시
        
    stats_features = np.array([
        f_min, f_max, f_mean, f_std, f_var, f_median, f_range, 
        f_cv, f_skew, f_kurt, f_q1, f_q2, f_q3, f_iqr, f_mcr, 
        f_abs_area, f_entropy
    ])
    
    # 2. Intensity Rhythm (CDP)
    # 논문은 CDP-N 파라미터를 중요하게 언급. 100Hz 데이터이므로 N=20 정도로 상세하게 봄.
    cdp_features = calculate_cdp(combined_signal, n_bins=20)
    
    # 최종 피처 벡터 결합
    return np.concatenate([stats_features, cdp_features])

from scipy.signal import periodogram # entropy 계산용

# --- 3. 데이터 로더 ---

def load_and_preprocess_paper(csv_file, target_fs=100.0):
    try:
        df = pd.read_csv(csv_file)
        df.replace('########', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if len(df) < 10: return None, None

        # 타입 변환 및 정렬
        df = df.astype({'timestamp': float, 'ax': float, 'ay': float, 'az': float, 
                        'gx': float, 'gy': float, 'gz': float, 'label': int})
        df.sort_values(by='timestamp', inplace=True)

        # 리샘플링 (100Hz)
        cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        t_orig = df['timestamp'].values
        
        # 중복 타임스탬프 제거
        _, u_idx = np.unique(t_orig, return_index=True)
        t_orig = t_orig[u_idx]
        data_orig = df[cols].values[u_idx]
        
        if len(t_orig) < 2: return None, None
        
        t_new = np.arange(t_orig[0], t_orig[-1], 1.0/target_fs)
        if len(t_new) < 2: return None, None
        
        resampled_data = np.zeros((len(t_new), len(cols)))
        for i in range(len(cols)):
            f_int = interp1d(t_orig, data_orig[:,i], kind='linear', fill_value="extrapolate")
            resampled_data[:,i] = f_int(t_new)
            
        # 1차 필터링 (노이즈 제거)
        filtered_data = lowpass_filter(resampled_data, fs=target_fs, cutoff=10)
        
        # DataFrame으로 다시 변환 (combine_axes 함수 사용을 위해)
        df_proc = pd.DataFrame(filtered_data, columns=cols)
        
        # 논문에서 언급된 방식의 축 결합 (Acc, Gyro 각각 적용 후 합치거나, Acc만 사용)
        # 논문은 Accelerometer만 사용했음. 여기서는 Acc 데이터만으로 피처를 추출해봄.
        # (Gyro도 같은 방식으로 처리해서 피처를 2배로 늘려도 됨)
        combined_acc = combine_axes_paper_method(df_proc[['ax', 'ay', 'az']])
        
        # Gyro도 아까우니 똑같이 처리해서 피처에 추가하자 (논문 확장)
        combined_gyro = combine_axes_paper_method(df_proc[['gx', 'gy', 'gz']].rename(columns={'gx':'ax', 'gy':'ay', 'gz':'az'}))
        
        label = df['label'].iloc[0]
        
        return combined_acc, combined_gyro, int(label)

    except Exception as e:
        print(f"Err: {csv_file} - {e}")
        return None, None, None

# --- 메인 실행 ---

TARGET_FS = 100.0
data_features = []
labels_list = []

print("Loading & Processing with Paper Methodology...")
csv_files = glob.glob("data/*.csv")

for f in csv_files:
    # Acc 결합신호, Gyro 결합신호, 라벨
    acc_comb, gyro_comb, label = load_and_preprocess_paper(f, target_fs=TARGET_FS)
    
    if acc_comb is not None:
        # 피처 추출 (Acc + Gyro 각각 추출 후 결합)
        # 논문 피처 17개 + CDP 20개 = 37개
        # Acc 37개 + Gyro 37개 = 총 74개 피처
        f_acc = extract_features_paper(acc_comb, fs=TARGET_FS)
        f_gyro = extract_features_paper(gyro_comb, fs=TARGET_FS)
        
        final_features = np.concatenate([f_acc, f_gyro])
        
        data_features.append(final_features)
        labels_list.append(label)

print(f"Processed {len(data_features)} samples.")

if not data_features:
    print("No data found.")
    exit()

# --- 데이터 불균형 처리 ---
cnt = Counter(labels_list)
valid_labels = {l for l, c in cnt.items() if c >= 2}
if len(valid_labels) < len(cnt):
    print(f"Filtering labels < 2 samples...")
    data_features, labels_list = zip(*[(d, l) for d, l in zip(data_features, labels_list) if l in valid_labels])

# --- 훈련/테스트 분리 ---
X_train, X_test, y_train, y_test = train_test_split(
    data_features, labels_list, test_size=0.2, random_state=42, stratify=labels_list
)

# --- 표준화 ---
# 피처 단위가 다르므로 (Entropy vs Area) 표준화 필수
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 모델 학습 (Random Forest) ---
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# --- 평가 ---
y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n=== Results (Paper Method) ===")
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
u_labels = sorted(np.unique(y_test))
t_labels = [WORD_CLASSES[i] for i in u_labels if i < len(WORD_CLASSES)]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=t_labels, yticklabels=t_labels)
plt.title('Confusion Matrix (Rhythm Attack Method)')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout(); plt.show()
