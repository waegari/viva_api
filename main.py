import os
import cv2
import json
import torch
import numpy as np
import tempfile
import shutil
import replicate
import Levenshtein  # pip install python-levenshtein
import torch.nn as nn
import mediapipe as mp
from contextlib import asynccontextmanager
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fastapi import FastAPI, HTTPException, Body, Query, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv # pip install python-dotenv

# 0. 환경 변수 로드 (보안)
load_dotenv()  # .env 파일에서 REPLICATE_API_TOKEN을 불러옵니다.

# ==========================================================
# 1. 데이터 모델 (요청 및 응답 정의) - Swagger 표시용
# ==========================================================

# [요청] 랜드마크 데이터
class LandmarkPos(BaseModel):
    x: float; y: float; z: float

class FaceTrackingData(BaseModel):
    seg_key: str; label: str; frame_idx: int; time_sec: float
    landmarks: Dict[str, LandmarkPos]

# [응답] 분석 결과 (Swagger에 이 구조가 그대로 뜸)
class AnalysisResult(BaseModel):
    status: str
    target_word: str
    label_in_file: Optional[str] = None
    similarity: Optional[float] = None
    score: float
    passed: bool
    top_error_landmarks: Optional[List[int]] = []
    recognized_text: Optional[str] = None  # 오디오 분석용

# ==========================================================
# 2. 모델 아키텍처 & 유틸리티
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
# 3. 통합 엔진 (비디오 + JSON + 오디오)
# ==========================================================
class UnifiedEvaluator:
    def __init__(self, ckpt_path, ref_files: Dict[str, str], device="cpu"):
        self.device = device
        print(f"[Init] Loading Model: {ckpt_path}")
        
        # 입모양 모델 로드
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = EncoderAttn(ck["feat_dim"], ck["hidden"], ck["layers"], attn_dim=ck["attn_dim"])
        self.model.load_state_dict(ck["state"]); self.model.to(device).eval()
        
        self.mean = ck["mean"]; self.std = ck["std"]; self.std_raw = ck["std_raw"]
        self.use_delta = ck["use_delta"]; self.lip_indices = ck.get("lip_indices")
        self.idx_61 = self.lip_indices.index(61); self.idx_291 = self.lip_indices.index(291)

        # 레퍼런스 파일들 로드
        self.refs = {}
        for key, path in ref_files.items():
            if os.path.exists(path):
                rf = np.load(path)
                # 파일 구조 유연하게 처리 (단일/멀티 키)
                if f"{key}_proto" in rf:
                    self.refs[key] = {"proto": rf[f"{key}_proto"], "rep_seq": rf[f"{key}_rep_seq"], "label": key}
                elif "proto" in rf:
                    self.refs[key] = {"proto": rf["proto"], "rep_seq": rf["rep_seq"], "label": str(rf["label"])}
                print(f"   ✅ Loaded Reference: {key}")

        # 비디오 처리용 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3
        )

    # --- [A] 비디오 처리 ---
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        seq = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                all_xyz = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                if all_xyz.shape[0] >= 468:
                    pts = all_xyz[self.lip_indices]
                    seq.append(normalize_points_dynamic(pts, self.idx_61, self.idx_291))
            else:
                if seq: seq.append(seq[-1])
        cap.release()
        return np.array(seq, dtype=np.float32) if seq else None

    # --- [B] JSON 처리 ---
    def process_json(self, data_list: List[FaceTrackingData]):
        seq = []
        for frame in data_list:
            lms = frame.landmarks
            if not lms: continue
            pts = []
            valid = True
            for idx in self.lip_indices:
                key = str(idx)
                if key not in lms: valid = False; break
                pts.append([lms[key].x, lms[key].y, lms[key].z])
            if valid:
                pts_arr = np.array(pts, dtype=np.float32)
                seq.append(normalize_points_dynamic(pts_arr, self.idx_61, self.idx_291))
        return np.array(seq, dtype=np.float32) if seq else None

    # --- [C] 입모양 평가 로직 ---
    def evaluate_lip(self, seq, target_key: str):
        if target_key not in self.refs:
             # 테스트를 위해 파일이 없어도 'korean' 데이터로 강제 매핑 (임시)
             # 실제 배포시에는 에러를 띄우거나 정확한 파일 필요
             if 'korean' in self.refs: ref_data = self.refs['korean'] 
             else: return {"status": "error", "message": "Ref not found"}
        else:
            ref_data = self.refs[target_key]

        if seq is None or len(seq) < 5:
            return {"status": "fail", "message": "Video/Data too short"}

        x = flatten_xyz(seq)
        if self.use_delta: x = add_delta(x)
        x = (x - self.mean) / (self.std + 1e-6)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        lt = torch.tensor([x.shape[0]]).to(self.device)
        
        with torch.no_grad():
            z, alphas = self.model.forward_embed(xt, lt, return_alpha=True)
            z = z.cpu().numpy()[0]; alpha = alphas[0].cpu().numpy()

        nz = z / (np.linalg.norm(z)+1e-9)
        np_ = ref_data["proto"] / (np.linalg.norm(ref_data["proto"])+1e-9)
        cos_sim = float(nz @ np_)
        
        score = max(0.0, min(100.0, (cos_sim - 0.92) / (0.99 - 0.92) * 100.0)) if cos_sim >= 0.92 else 0.0

        diffs = dtw_diff(seq, ref_data["rep_seq"], std=self.std_raw)
        alpha_np = alpha[:, None]
        point_scores = (diffs * alpha_np).sum(axis=0) / (alpha_np.sum() + 1e-9)
        top_errors = np.argsort(-point_scores)[:5]

        return {
            "status": "success",
            "target_word": target_key,
            "label_in_file": ref_data["label"],
            "similarity": cos_sim,
            "score": round(score, 2),
            "passed": score >= 80.0,
            "top_error_landmarks": [int(self.lip_indices[i]) for i in top_errors]
        }
    
    # --- [D] 오디오 평가 (CER) ---
    def evaluate_audio(self, audio_path, target_text):
        # 1. Replicate Whisper 호출
        try:
            output = replicate.run(
                "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
                input={"audio": open(audio_path, "rb"), "language": "ko"}
            )
            # output format check (Replicate 버전에 따라 다를 수 있음)
            pred_text = output.get("transcription", "") if isinstance(output, dict) else str(output)
        except Exception as e:
            return {"status": "error", "message": f"Replicate Error: {str(e)}", "score": 0, "passed": False}

        # 2. 공백 제거 및 전처리
        def clean(s): return s.replace(" ", "").strip()
        
        ref_clean = clean(target_text)
        pred_clean = clean(pred_text)
        
        # 3. CER 계산 (Levenshtein Distance / Len)
        if len(ref_clean) == 0:
            cer = 1.0 if len(pred_clean) > 0 else 0.0
        else:
            dist = Levenshtein.distance(ref_clean, pred_clean)
            cer = dist / len(ref_clean)
        
        # 4. 점수 환산 (CER 0 -> 100점)
        score = max(0.0, (1.0 - cer) * 100.0)
        
        return {
            "status": "success",
            "target_word": target_text,
            "recognized_text": pred_text, # 인식된 원본 텍스트
            "score": round(score, 2),
            "passed": score >= 80.0 # 기준은 자유롭게 설정
        }


# ==========================================================
# 4. FastAPI 앱
# ==========================================================
CKPT_PATH = "ckpt.pt"
# [임시] 모든 키를 refpack.npz로 연결 (테스트용)
REF_FILES_MAP = {
    "korean": "refpack.npz",
    "read": "refpack.npz", 
    "rice": "refpack.npz"
}

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if os.path.exists(CKPT_PATH):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            engine = UnifiedEvaluator(CKPT_PATH, REF_FILES_MAP, device=device)
            print("✅ Server Engine Ready!")
        except Exception as e:
            print(f"❌ Engine Init Error: {e}")
    yield
    if engine: engine.mp_face_mesh.close()

app = FastAPI(lifespan=lifespan)

# [1] 비디오 분석 (테스트용)
@app.post("/analyze/video", response_model=AnalysisResult)
async def analyze_video(
    target: str = Query(..., description="Target word (korean, rice, read.)"),
    file: UploadFile = File(...)
):
    if not engine: raise HTTPException(503, "Loading...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp); tmp_path = tmp.name
    try:
        seq = engine.process_video(tmp_path)
        res = engine.evaluate_lip(seq, target)
        if res["status"] == "error": raise HTTPException(400, res["message"])
        return res
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# [2] JSON 분석 (실서비스용)
@app.post("/analyze/json", response_model=AnalysisResult)
async def analyze_json(
    target: str = Query(..., description="Target word"),
    data: List[FaceTrackingData] = Body(...)
):
    if not engine: raise HTTPException(503, "Loading...")
    seq = engine.process_json(data)
    res = engine.evaluate_lip(seq, target)
    if res["status"] == "error": raise HTTPException(400, res["message"])
    return res

# [3] 오디오 분석 (New!)
@app.post("/analyze/audio", response_model=AnalysisResult)
async def analyze_audio(
    target_text: str = Form(..., description="Expected text (e.g. 안녕하세요)"),
    file: UploadFile = File(...)
):
    """
    오디오 파일을 업로드하여 STT(Replicate) 후 발음 정확도(CER) 점수 반환
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HTTPException(500, "Server Error: Replicate API Token not configured.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp); tmp_path = tmp.name
    
    try:
        # Replicate API 호출이 있으므로 약간의 시간이 소요됨
        res = engine.evaluate_audio(tmp_path, target_text)
        
        if res["status"] == "error":
            raise HTTPException(500, res["message"])
            
        # Pydantic 모델에 맞춰 불필요한 필드 채우기
        return {
            "status": "success",
            "target_word": target_text,
            "score": res["score"],
            "passed": res["passed"],
            "recognized_text": res["recognized_text"],
            "label_in_file": target_text, # 오디오는 라벨과 타겟이 같음
            "similarity": 0.0, # 오디오에는 해당 없음
            "top_error_landmarks": [] # 오디오에는 해당 없음
        }
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)