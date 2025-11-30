import os
import torch
import numpy as np
import tempfile
import shutil
import replicate
import Levenshtein 
import torch.nn as nn
import mediapipe as mp
import importlib.util
from contextlib import asynccontextmanager
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fastapi import FastAPI, HTTPException, Body, Query, File, UploadFile, Form, Path
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv 

# 0. 환경 변수 로드
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_ROOT = os.path.join(BASE_DIR, "infer")

# [호환성 레이어] v1(영어 키) -> v2(한글 라벨) 매핑
# 클라이언트가 v1 방식으로 'korean'을 요청하면, 내부적으로 '한국어' 모델을 찾아갑니다.
LEGACY_MAPPING = {
    "korean": "한국어",
    "read": "읽어요",
    "bab": "볶음밥",
    # 필요한 경우 여기에 예전 키들을 더 추가하세요.
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
    target_id: str
    label_in_file: Optional[str]
    similarity: Optional[float] = None
    score: float
    passed: bool
    top_error_landmarks: Optional[List[int]] = []
    recognized_text: Optional[str] = None 

# ==========================================================
# 2. 오디오 평가 함수
# ==========================================================
def process_audio_scoring(audio_path: str, target_text: str) -> dict:
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN is missing")

    try:
        with open(audio_path, "rb") as audio_file:
            output = replicate.run(
                "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
                input={"audio": audio_file, "language": "ko", "model": "large-v3"}
            )
        pred_text = output.get("transcription", "") if isinstance(output, dict) else str(output)
    except Exception as e:
        print(f"Replicate Error: {e}")
        return {"status": "error", "message": str(e), "score": 0.0, "passed": False}

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
    def __init__(self, root_dir: str, device="cpu"):
        self.device = device
        self.engines = {}
        self.label_to_id = {} 
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3
        )

        py_path = os.path.join(root_dir, "labels.py")
        print(f"[Init] Loading configuration from '{py_path}' ...")

        if not os.path.exists(py_path):
            print("⚠️ Critical Error: 'labels.py' not found.")
            return

        try:
            spec = importlib.util.spec_from_file_location("config_labels", py_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            if not hasattr(config_module, "labels"):
                raise ValueError("Variable 'labels' not found in labels.py")
                
            self.mapping = config_module.labels
            self.label_to_id = {v: k for k, v in self.mapping.items()}
            print(f" ℹ️ Parsed {len(self.mapping)} labels.")
            
        except Exception as e:
            print(f"❌ Python Import Error: {e}")
            return

        loaded_count = 0
        
        for model_id, label_text in self.mapping.items():
            base_path = os.path.join(root_dir, str(model_id))
            c_path = os.path.join(base_path, "ckpt.pt")
            r_path = os.path.join(base_path, "refpack.npz")

            if not (os.path.exists(c_path) and os.path.exists(r_path)):
                continue

            try:
                ck = torch.load(c_path, map_location=device, weights_only=False)
                model = EncoderAttn(ck["feat_dim"], ck["hidden"], ck["layers"], attn_dim=ck["attn_dim"])
                model.load_state_dict(ck["state"]); model.to(device).eval()
                
                rf = np.load(r_path)
                keys = list(rf.keys())
                proto = rf[keys[0]]; rep_seq = rf[keys[1]]
                for k in keys:
                    if "proto" in k: proto = rf[k]
                    if "rep_seq" in k: rep_seq = rf[k]

                self.engines[str(model_id)] = {
                    "model": model, "proto": proto, "rep_seq": rep_seq, 
                    "label": label_text,
                    "mean": ck["mean"], "std": ck["std"], "std_raw": ck["std_raw"], 
                    "use_delta": ck["use_delta"],
                    "lip_indices": ck.get("lip_indices"),
                    "idx_61": ck.get("lip_indices").index(61), 
                    "idx_291": ck.get("lip_indices").index(291)
                }
                loaded_count += 1
            except Exception as e:
                print(f" ❌ Error loading ID '{model_id}': {e}")
        
        print(f" ✅ Successfully loaded {loaded_count} models.\n")

    def resolve_id(self, identifier: str) -> Optional[str]:
        # 1. ID 직접 매칭
        if identifier in self.engines: return identifier
        # 2. 라벨(한글) -> ID 매칭
        if identifier in self.label_to_id:
            found = self.label_to_id[identifier]
            if found in self.engines: return found
        # 3. Legacy 키워드(korean) -> 라벨(한국어) -> ID 매칭
        if identifier in LEGACY_MAPPING:
            converted_label = LEGACY_MAPPING[identifier]
            if converted_label in self.label_to_id:
                found = self.label_to_id[converted_label]
                if found in self.engines: return found
        return None

    def get_label(self, model_id: str) -> Optional[str]:
        if model_id in self.engines: return self.engines[model_id]["label"]
        return None

    def process_json(self, data_list: List[FaceTrackingData], model_id: str):
        if model_id not in self.engines: return None
        eng = self.engines[model_id]
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

    def evaluate_lip(self, seq, model_id: str):
        if model_id not in self.engines:
            return {"status": "error", "message": f"Model ID '{model_id}' not found."}
        
        eng = self.engines[model_id]
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
        
        diffs = dtw_diff(seq, eng["rep_seq"], std=eng["std_raw"])
        mean_diff = diffs.mean()
        
        penalty = max(0.0, (mean_diff - 1.5) * 20.0)
        base_score = (cos_sim - 0.90) / (0.99 - 0.90) * 100.0
        score = max(0.0, min(100.0, base_score - penalty))

        alpha_np = alpha[:, None]
        point_scores = (diffs * alpha_np).sum(axis=0) / (alpha_np.sum() + 1e-9)
        top_errors = np.argsort(-point_scores)[:5]

        return {
            "status": "success",
            "target_id": model_id,
            "label_in_file": eng["label"],
            "similarity": cos_sim,
            "score": round(score, 2),
            "passed": score >= 80.0,
            "top_error_landmarks": [int(eng["lip_indices"][i]) for i in top_errors]
        }

# ==========================================================
# 5. FastAPI 앱 실행 및 라우팅 (v1, v2 분리)
# ==========================================================
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = MultiModelEvaluator(INFER_ROOT, device=device)
    except Exception as e:
        print(f"❌ Engine Init Error: {e}")
    yield
    if engine: engine.mp_face_mesh.close()

app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------------------
# [v2] New API (Path Parameter 방식: /analyze/json/1, /analyze/json/가수)
# ------------------------------------------------------------------
@app.post("/analyze/audio/{identifier}", response_model=AnalysisResult)
async def analyze_audio_v2(
    identifier: str = Path(..., description="모델 ID ('1') 또는 라벨 ('가수')"),
    file: UploadFile = File(...)
):
    if not engine: raise HTTPException(503, "Engine not loaded")
    
    real_id = engine.resolve_id(identifier)
    if not real_id:
        raise HTTPException(404, f"Model identifier '{identifier}' not found.")
    
    target_text = engine.get_label(real_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp); tmp_path = tmp.name
    try:
        res = process_audio_scoring(tmp_path, target_text)
        if res["status"] == "error": raise HTTPException(500, res["message"])
        return {
            "status": "success", "target_id": real_id, "label_in_file": target_text,
            "score": res["score"], "passed": res["passed"], "recognized_text": res["recognized_text"],
            "similarity": 0.0, "top_error_landmarks": []
        }
    except RuntimeError as e: raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

@app.post("/analyze/json/{identifier}", response_model=AnalysisResult)
async def analyze_json_v2(
    identifier: str = Path(..., description="모델 ID ('1') 또는 라벨 ('가수')"),
    data: List[FaceTrackingData] = Body(...)
):
    if not engine: raise HTTPException(503, "Engine not loaded")
    real_id = engine.resolve_id(identifier)
    if not real_id: raise HTTPException(404, f"Model identifier '{identifier}' not found")

    seq = engine.process_json(data, real_id)
    res = engine.evaluate_lip(seq, real_id)
    if res.get("status") == "error": raise HTTPException(400, res["message"])
    return res


# ------------------------------------------------------------------
# [v1] Legacy API (Query Parameter 방식: /analyze/json?target=korean)
# ------------------------------------------------------------------
@app.post("/analyze/audio", response_model=AnalysisResult)
async def analyze_audio_v1(
    target: str = Form(..., description="Legacy target (e.g., korean)"),
    file: UploadFile = File(...)
):
    # v1 요청을 v2 로직으로 우회 처리 (resolve_id가 매핑 처리함)
    return await analyze_audio_v2(identifier=target, file=file)

@app.post("/analyze/json", response_model=AnalysisResult)
async def analyze_json_v1(
    target: str = Query(..., description="Legacy target (e.g., korean)"),
    data: List[FaceTrackingData] = Body(...)
):
    # v1 요청을 v2 로직으로 우회 처리
    return await analyze_json_v2(identifier=target, data=data)

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(INFER_ROOT):
        os.makedirs(INFER_ROOT, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)