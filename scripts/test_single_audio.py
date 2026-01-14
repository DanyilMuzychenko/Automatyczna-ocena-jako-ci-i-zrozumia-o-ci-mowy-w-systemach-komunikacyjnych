# =========================
# TEST SINGLE AUDIO FILE
# =========================
import os
import sys
import torch
import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# =========================
# IMPORTS
# =========================
from models.nisqa_cnn import NISQACNN
from models.nisqa_cnn_gru import NISQACNN_GRU
from models.nisqa_cnn_lstm import NISQACNN_LSTM
from models.nisqa_wav2vec_rnn import NISQAWav2VecRNN
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
DURATION = 5.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION)


# =========================
# FEATURE EXTRACTION FUNCTIONS
# =========================
def load_and_preprocess_audio(audio_path, target_length=MAX_SAMPLES):
    waveform, sr = torchaudio.load(audio_path)


    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)


    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)


    current_length = waveform.shape[-1]
    if current_length > target_length:
        waveform = waveform[..., :target_length]
    elif current_length < target_length:
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    return waveform


def extract_mel(audio_path):
    waveform = load_and_preprocess_audio(audio_path)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=64
    )
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-9)

    return mel  # shape: [1, 64, T]


def extract_mfcc(audio_path):
    waveform = load_and_preprocess_audio(audio_path)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=20,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 64,
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc  # shape: [1, 20, T]


def extract_wav2vec2(audio_path):
    waveform = load_and_preprocess_audio(audio_path)


    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)  # [1, samples] -> [samples]

    print(f"[DEBUG] Waveform shape for wav2vec2: {waveform.shape}")


    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()


    inputs = processor(waveform, sampling_rate=SAMPLE_RATE,
                       return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state  # shape: [1, T, 768]

    return features.squeeze(0)  # shape: [T, 768]


# =========================
# MOS DENORMALIZATION
# =========================
def denormalize_mos(mos_norm):
    return mos_norm * 4.0 + 1.0


# =========================
# USER INPUT
# =========================
print("\nSelect input features:")
print("1 - Mel-spectrogram")
print("2 - MFCC")
print("3 - wav2vec2")
feat_choice = input("Choice [1/2/3]: ").strip()

print("\nSelect model:")
if feat_choice == "3":
    print("1 - wav2vec2 + GRU")
    print("2 - wav2vec2 + LSTM")
else:
    print("1 - CNN")
    print("2 - CNN + GRU")
    print("3 - CNN + LSTM")

model_choice = input("Choice: ").strip()

audio_path = input("\nPath to WAV file: ").strip()
if not os.path.isfile(audio_path):
    print(f"[ERROR] Audio file not found: {audio_path}")
    sys.exit(1)

# =========================
# LOAD FEATURES
# =========================
print("\n[INFO] Extracting features...")
try:
    if feat_choice == "1":
        x = extract_mel(audio_path)
        model_type = "mel"
        # Add channel dimension for CNN: [1, 64, T] -> [1, 1, 64, T]
        x = x.unsqueeze(0)  # Add batch dimension
        print(f"[INFO] Mel shape after unsqueeze: {x.shape}")

    elif feat_choice == "2":
        x = extract_mfcc(audio_path)
        model_type = "mfcc"
        # Add channel dimension for CNN: [1, 20, T] -> [1, 1, 20, T]
        x = x.unsqueeze(0)  # Add batch dimension
        print(f"[INFO] MFCC shape after unsqueeze: {x.shape}")

    elif feat_choice == "3":
        x = extract_wav2vec2(audio_path)
        model_type = "wav2vec2"
        # wav2vec2: [T, 768] -> [1, T, 768]
        x = x.unsqueeze(0)  # Add batch dimension
        print(f"[INFO] Wav2vec2 shape after unsqueeze: {x.shape}")

    else:
        raise ValueError("Invalid feature choice")

    x = x.to(DEVICE)
    print(f"[INFO] Final tensor shape: {x.shape}")

except Exception as e:
    print(f"[ERROR] Feature extraction failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# =========================
# LOAD MODEL
# =========================
print("\n[INFO] Loading model...")
try:
    # Check if model weights exist
    if model_type == "wav2vec2":
        if model_choice == "1":
            model = NISQAWav2VecRNN("gru")
            weights_dir = "outputs/WAV2VEC_GRU_wav2vec2"
        else:
            model = NISQAWav2VecRNN("lstm")
            weights_dir = "outputs/WAV2VEC_LSTM_wav2vec2"
    else:
        if model_choice == "1":
            model = NISQACNN()
            weights_dir = f"outputs/CNN_{model_type}"
        elif model_choice == "2":
            model = NISQACNN_GRU()
            weights_dir = f"outputs/CNN_GRU_{model_type}"
        else:
            model = NISQACNN_LSTM()
            weights_dir = f"outputs/CNN_LSTM_{model_type}"

    weights_path = os.path.join(weights_dir, "model.pt")

    if not os.path.exists(weights_path):
        print(f"[ERROR] Model weights not found: {weights_path}")
        print("Available outputs directories:")
        if os.path.exists("outputs"):
            for item in os.listdir("outputs"):
                print(f"  - {item}")
        sys.exit(1)

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded from: {weights_path}")

    # Print model architecture for debugging
    print(f"[DEBUG] Model type: {type(model)}")

except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# =========================
# PREDICTION
# =========================
print("\n[INFO] Running prediction...")
try:
    with torch.no_grad():
        print(f"[DEBUG] Input shape to model: {x.shape}")
        pred_norm = model(x)

        if isinstance(pred_norm, torch.Tensor):
            pred_norm = pred_norm.item()

        pred_mos = denormalize_mos(pred_norm)

    print("\n" + "=" * 40)
    print(f" Predicted MOS: {pred_mos:.2f} / 5.00")


    if pred_mos >= 4.0:
        quality = "Excellent"
    elif pred_mos >= 3.0:
        quality = "Good"
    elif pred_mos >= 2.0:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f" Quality: {quality}")
    print("=" * 40)

except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    print(f"[DEBUG] Model architecture: {model}")
    print(f"[DEBUG] Input dtype: {x.dtype}, device: {x.device}")
    import traceback

    traceback.print_exc()