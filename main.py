import os
import cv2
import json
import torch
import numpy as np
import tempfile
import shutil
import replicate
import Levenshtein 
import torch.nn as nn
import mediapipe as mp
from contextlib import asynccontextmanager
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fastapi import FastAPI, HTTPException, Body, Query, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv 

# 0. 환경 변수 로드
load_dotenv()

# ==========================================================
# [설정] 매핑 정보 (비디오/JSON용 + 오디오용)
# ==========================================================

# 1. 비디오/JSON 분석용 (모델 & 기준파일)
CKPT_PATH_MAP = {
    "korean": "ckpt_korean.pt",
    "read": "ckpt_read.pt",
    "bab": "ckpt_bab.pt",
}

REF_FILES_MAP = {
    "korean": "refpack_korean.npz",
    "read": "refpack_read.npz", 
    "bab": "refpack_bab.npz"
}

# 2. [New] 오디오 분석용 (Key -> 실제 정답 텍스트)
TARGET_TEXT_MAP = {
    "korean": "한국어",
    "read": "읽어요",
    "bab": "볶음밥",
}

# ==========================================================
# 1. 데이터 모델
# ==========================================================
class LandmarkPos(BaseModel):
    x: float; y: float; z: float

class FaceTrackingData(BaseModel):
    seg_key: str; label: str; frame_idx: int; time_sec: float
    landmarks: Dict[str, LandmarkPos]

class AnalysisResult(BaseModel):
    status: str
    target_word: str                 # 클라이언트가 보낸 키 (예: "korean")
    label_in_file: Optional[str] = None # 실제 매핑된 값 (예: "한국어")
    similarity: Optional[float] = None
    score: float
    passed: bool
    top_error_landmarks: Optional[List[int]] = []
    recognized_text: Optional[str] = None 

# ==========================================================
# 2. 오디오 평가 함수 (순수 로직)
# ==========================================================
def process_audio_scoring(audio_path: str, target_text: str) -> dict:
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN is missing in .env file")

    try:
        # [수정됨] 파일을 with 문으로 열어서, 사용 후 즉시 닫히도록 보장함 (WinError 32 해결 핵심)
        with open(audio_path, "rb") as audio_file:
            output = replicate.run(
                "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
                input={
                    "audio": audio_file,  # 열린 파일 핸들 전달
                    "language": "ko", 
                    "model": "large-v3"
                }
            )
        # with 블록을 나오면 파일이 자동으로 닫힘 -> os.remove 가능해짐
        
        pred_text = output.get("transcription", "") if isinstance(output, dict) else str(output)
        
    except Exception as e:
        print(f"Replicate Error: {e}")
        return {"status": "error", "message": str(e), "score": 0.0, "passed": False}

    # ... (아래 점수 계산 로직은 그대로 유지) ...
    def clean(s): return str(s).replace(" ", "").strip()
    ref_clean = clean(target_text)
    pred_clean = clean(pred_text)
    
    if len(ref_clean) == 0:
        cer = 1.0 if len(pred_clean) > 0 else 0.0
    else:
        dist = Levenshtein.distance(ref_clean, pred_clean)
        cer = dist / len(ref_clean)
    
    score = max(0.0, (1.0 - cer) * 100.0)
    
    return {
        "status": "success",
        "target_word": target_text,
        "recognized_text": pred_text,
        "score": round(score, 2),
        "passed": score >= 80.0
    }

# ==========================================================
# 3. 입모양 모델 아키텍처 & 유틸
# ==========================================================
class EncoderAttn(nn.Module):
    def __init__(self, d_in, h=256, layers=2, drop=0.3, attn_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, num_layers=layers, batch_first=True, dropout=drop, bidirectional=True)
        self.attn1 = nn.Linear(2*h, attn_dim); self.attn2 = nn.Linear(attn_dim, 1)
        self.ln = nn.LayerNorm(2*h)
    def forward_embed(self, x, lens, return_alpha=False):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        mask = (torch.arange(out.shape[1], device=out.device)[None,:] < lens[:,None])
        score = self.attn2(torch.tanh(self.attn1(out))).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1)
        z = self.ln((alpha.unsqueeze(-1) * out).sum(dim=1))
        return (z, alpha) if return_alpha else z

def normalize_points_dynamic(points, idx_61, idx_291):
    pL = points[idx_61, :2]; pR = points[idx_291, :2]
    center = (pL + pR) / 2.0
    scale = float(np.linalg.norm(pR - pL) + 1e-9)
    out = points.copy()
    out[:, 0:2] = (out[:, 0:2] - center) / scale
    out[:, 2]   = out[:, 2] / scale
    return out

def flatten_xyz(a): return a.reshape(a.shape[0], -1).astype(np.float32)
def add_delta(x): return np.concatenate([x, np.diff(x, axis=0, prepend=x[:1])], axis=1).astype(np.float32)

def dtw_diff(seq_q, seq_ref, std=None):
    A = flatten_xyz(seq_q); B = flatten_xyz(seq_ref)
    Ta, Tb = A.shape[0], B.shape[0]
    if Ta != Tb:
        x_new = np.linspace(0, 1, Ta); x_old = np.linspace(0, 1, Tb)
        B_res = np.zeros_like(A)
        for d in range(B.shape[1]): B_res[:,d] = np.interp(x_new, x_old, B[:,d])
        B = B_res
    diffs = np.linalg.norm(A.reshape(Ta,-1,3) - B.reshape(Ta,-1,3), axis=2)
    if std is not None: 
        std_p = np.linalg.norm(std.reshape(-1,3), axis=1) + 1e-6
        diffs /= std_p[None, :]
    return diffs

# ==========================================================
# 4. 멀티 모델 엔진
# ==========================================================
class MultiModelEvaluator:
    def __init__(self, ckpt_map: Dict[str, str], ref_map: Dict[str, str], device="cpu"):
        self.device = device
        self.engines = {}
        
        print("[Init] Loading Models & References...")
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3
        )

        targets = set(ckpt_map.keys()).intersection(set(ref_map.keys()))
        
        for key in targets:
            c_path = ckpt_map[key]
            r_path = ref_map[key]
            
            if not os.path.exists(c_path) or not os.path.exists(r_path):
                print(f"   ⚠️  Files missing for '{key}' (Skip)")
                continue
                
            try:
                ck = torch.load(c_path, map_location=device, weights_only=False)
                model = EncoderAttn(ck["feat_dim"], ck["hidden"], ck["layers"], attn_dim=ck["attn_dim"])
                model.load_state_dict(ck["state"]); model.to(device).eval()
                
                rf = np.load(r_path)
                if f"{key}_proto" in rf:
                    proto = rf[f"{key}_proto"]; rep_seq = rf[f"{key}_rep_seq"]; label = str(key)
                else:
                    proto = rf["proto"]; rep_seq = rf["rep_seq"]; label = str(rf["label"]) if "label" in rf else key

                self.engines[key] = {
                    "model": model, "proto": proto, "rep_seq": rep_seq, "label": label,
                    "mean": ck["mean"], "std": ck["std"], "std_raw": ck["std_raw"], "use_delta": ck["use_delta"],
                    "lip_indices": ck.get("lip_indices"),
                    "idx_61": ck.get("lip_indices").index(61), "idx_291": ck.get("lip_indices").index(291)
                }
                print(f"   ✅ Loaded Target '{key}'")
            except Exception as e:
                print(f"   ❌ Error loading '{key}': {e}")

    def process_json(self, data_list: List[FaceTrackingData], target_key: str):
        if target_key not in self.engines: return None
        eng = self.engines[target_key]
        
        seq = []
        for frame in data_list:
            lms = frame.landmarks
            if not lms: continue
            pts = []
            valid = True
            for idx in eng["lip_indices"]:
                key = str(idx)
                if key not in lms: valid = False; break
                pts.append([lms[key].x, lms[key].y, lms[key].z])
            if valid:
                pts_arr = np.array(pts, dtype=np.float32)
                seq.append(normalize_points_dynamic(pts_arr, eng["idx_61"], eng["idx_291"]))
        return np.array(seq, dtype=np.float32) if seq else None

    def evaluate_lip(self, seq, target_key: str):
        if target_key not in self.engines:
            return {"status": "error", "message": f"Target '{target_key}' not loaded."}
        
        eng = self.engines[target_key]
        if seq is None or len(seq) < 5:
            return {"status": "fail", "message": "Input too short"}

        x = flatten_xyz(seq)
        if eng["use_delta"]: x = add_delta(x)
        x = (x - eng["mean"]) / (eng["std"] + 1e-6)
        
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        lt = torch.tensor([x.shape[0]]).to(self.device)
        
        with torch.no_grad():
            z, alphas = eng["model"].forward_embed(xt, lt, return_alpha=True)
            z = z.cpu().numpy()[0]; alpha = alphas[0].cpu().numpy()

        nz = z / (np.linalg.norm(z)+1e-9)
        np_ = eng["proto"] / (np.linalg.norm(eng["proto"])+1e-9)
        cos_sim = float(nz @ np_)
        
        # 개선된 점수 계산 (DTW 반영)
        diffs = dtw_diff(seq, eng["rep_seq"], std=eng["std_raw"])
        mean_diff = diffs.mean()
        
        penalty = max(0.0, (mean_diff - 1.5) * 20.0) # Penalty logic
        base_score = (cos_sim - 0.90) / (0.99 - 0.90) * 100.0
        score = max(0.0, min(100.0, base_score - penalty))

        alpha_np = alpha[:, None]
        point_scores = (diffs * alpha_np).sum(axis=0) / (alpha_np.sum() + 1e-9)
        top_errors = np.argsort(-point_scores)[:5]

        return {
            "status": "success",
            "target_word": target_key,
            "label_in_file": eng["label"],
            "similarity": cos_sim,
            "score": round(score, 2),
            "passed": score >= 80.0,
            "top_error_landmarks": [int(eng["lip_indices"][i]) for i in top_errors]
        }

# ==========================================================
# 5. FastAPI 앱 실행
# ==========================================================
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    available = any(os.path.exists(p) for p in CKPT_PATH_MAP.values())
    if available:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            engine = MultiModelEvaluator(CKPT_PATH_MAP, REF_FILES_MAP, device=device)
            print(f"✅ Server Ready! Targets: {list(engine.engines.keys())}")
        except Exception as e:
            print(f"❌ Engine Error: {e}")
    else:
        print("⚠️ No model files. Lip analysis disabled.")
    yield
    if engine: engine.mp_face_mesh.close()

app = FastAPI(lifespan=lifespan)

# [Endpoint] 오디오 (수정됨: Key -> Text 매핑 적용)
@app.post("/analyze/audio", response_model=AnalysisResult)
async def analyze_audio(
    target: str = Form(..., description="정답 텍스트 직접 입력 (예: 한국어, 볶음밥)"),
    file: UploadFile = File(...)
):
    # 2. 파일 저장 및 분석
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp); tmp_path = tmp.name
    try:
        res = process_audio_scoring(tmp_path, target)
        if res["status"] == "error": raise HTTPException(500, res["message"])
        
        # 3. 응답 생성 (Key와 실제 텍스트 모두 포함)
        return {
            "status": "success",
            "target_word": target,            # 클라이언트가 보낸 Key
            "label_in_file": target, # 실제 채점된 한글 텍스트
            "score": res["score"],
            "passed": res["passed"],
            "recognized_text": res["recognized_text"],
            "similarity": 0.0,
            "top_error_landmarks": []
        }
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# [Endpoint] JSON
@app.post("/analyze/json", response_model=AnalysisResult)
async def analyze_json(
    target: str = Query(..., description="korean, read, bab"),
    data: List[FaceTrackingData] = Body(...)
):
    if not engine: raise HTTPException(503, "Engine not loaded")
    seq = engine.process_json(data, target)
    res = engine.evaluate_lip(seq, target)
    if res.get("status") == "error": raise HTTPException(400, res["message"])
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)