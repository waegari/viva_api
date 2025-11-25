import os
import cv2
import json
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# ====================== Model Architecture ======================
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

# ====================== Helper Functions ======================
def normalize_points_dynamic(points, idx_61, idx_291):
    """
    points: (N, 3) array (selected points)
    idx_61: index of landmark 61 within the 'points' array
    idx_291: index of landmark 291 within the 'points' array
    """
    pL = points[idx_61, :2]
    pR = points[idx_291, :2]
    center = (pL + pR) / 2.0
    scale = float(np.linalg.norm(pR - pL) + 1e-9)
    out = points.copy()
    out[:, 0:2] = (out[:, 0:2] - center) / scale
    out[:, 2]   = out[:, 2] / scale
    return out

def flatten_xyz(a): return a.reshape(a.shape[0], -1).astype(np.float32)
def add_delta(x):
    dx = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, dx], axis=1).astype(np.float32)

def dtw_diff(seq_q, seq_ref, weights=None, std=None):
    # Simplified DTW for scoring (returns per-point diff map)
    A = flatten_xyz(seq_q); B = flatten_xyz(seq_ref)
    Ta, Tb = A.shape[0], B.shape[0]
    
    # Simple alignment
    if Ta != Tb:
        x_old = np.linspace(0, 1, Tb); x_new = np.linspace(0, 1, Ta)
        B_res = np.zeros_like(A)
        for d in range(B.shape[1]): B_res[:,d] = np.interp(x_new, x_old, B[:,d])
        B = B_res
    
    # Calculate Diff
    P = A.shape[1] // 3
    A_3d = A.reshape(Ta, P, 3); B_3d = B.reshape(Ta, P, 3)
    diffs = np.linalg.norm(A_3d - B_3d, axis=2) # (T, P)
    
    if weights is not None: diffs *= weights[None, :]
    if std is not None: 
        std_p = np.linalg.norm(std.reshape(-1,3), axis=1) + 1e-6
        diffs /= std_p[None, :]
        
    return diffs

# ====================== Inference Engine ======================
class PronunciationEvaluator:
    def __init__(self, ckpt_path, refpack_path, device="cpu"):
        self.device = device
        print(f"[Init] Loading model: {ckpt_path}")
        
        # Load Model & Meta (weights_only=False for PyTorch 2.6+)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        self.model = EncoderAttn(ck["feat_dim"], ck["hidden"], ck["layers"], attn_dim=ck["attn_dim"])
        self.model.load_state_dict(ck["state"])
        self.model.to(device).eval()
        
        self.mean = ck["mean"]
        self.std = ck["std"]
        self.std_raw = ck["std_raw"]
        self.use_delta = ck["use_delta"]
        
        # Load Lip Indices used during training
        self.lip_indices = ck.get("lip_indices")
        if self.lip_indices is None:
            raise ValueError("Checkpoint does not contain 'lip_indices'. Train with V3 code.")
            
        try:
            self.idx_61 = self.lip_indices.index(61)
            self.idx_291 = self.lip_indices.index(291)
        except ValueError:
            raise ValueError("Trained lip indices must contain 61 and 291 for normalization.")

        # Load Reference
        rf = np.load(refpack_path)
        self.proto = rf["proto"]
        self.rep_seq = rf["rep_seq"]
        self.label = str(rf["label"])
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, 
            refine_landmarks=False, min_detection_confidence=0.3
        )

    def process_json(self, json_data):
        """Input: List of dicts -> Output: Normalized Sequence"""
        if not json_data: return None
        seq = []
        for frame in json_data:
            lms = frame.get("landmarks", {})
            if not lms: continue
            pts = []
            valid = True
            for idx in self.lip_indices:
                key = str(idx)
                if key not in lms: valid = False; break
                pt = lms[key]
                pts.append([pt["x"], pt["y"], pt["z"]])
            if valid:
                pts_arr = np.array(pts, dtype=np.float32)
                pts_norm = normalize_points_dynamic(pts_arr, self.idx_61, self.idx_291)
                seq.append(pts_norm)
        return np.array(seq, dtype=np.float32) if seq else None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        seq = []; frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                all_xyz = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                if all_xyz.shape[0] >= 468:
                    pts = all_xyz[self.lip_indices]
                    pts_norm = normalize_points_dynamic(pts, self.idx_61, self.idx_291)
                    seq.append(pts_norm)
            else:
                if seq: seq.append(seq[-1])
        cap.release()
        return np.array(frames), np.array(seq, dtype=np.float32) if seq else None

    def evaluate(self, input_source, is_json=False, out_report=None):
        # 1. Prepare Data
        if is_json:
            if isinstance(input_source, str):
                with open(input_source, "r", encoding="utf-8") as f: data = json.load(f)
            else: data = input_source
            seq = self.process_json(data)
            frames = None
        else:
            frames, seq = self.process_video(input_source)

        if seq is None or len(seq) < 5:
            return {"status": "fail", "message": "Input too short or invalid"}

        # 2. Forward
        x = flatten_xyz(seq)
        if self.use_delta: x = add_delta(x)
        x = (x - self.mean) / (self.std + 1e-6)
        
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        lt = torch.tensor([x.shape[0]]).to(self.device)
        
        with torch.no_grad():
            z, alphas = self.model.forward_embed(xt, lt, return_alpha=True)
            z = z.cpu().numpy()[0]
            alpha = alphas[0].cpu().numpy() # Tensor -> Numpy conversion fixed

        # 3. Score Calculation (Strict Mode)
        nz = z / (np.linalg.norm(z)+1e-9)
        np_ = self.proto / (np.linalg.norm(self.proto)+1e-9)
        cos_sim = float(nz @ np_)
        
        # [STRICT THRESHOLD APPLIED]
        # 0.92 미만 -> 0점
        # 0.99 이상 -> 100점
        strict_low = 0.92
        strict_high = 0.99
        
        if cos_sim < strict_low:
            score = 0.0
        else:
            score = (cos_sim - strict_low) / (strict_high - strict_low) * 100.0
            score = max(0.0, min(100.0, score))

        # 4. Error Analysis
        diffs = dtw_diff(seq, self.rep_seq, std=self.std_raw)
        alpha_np = alpha[:, None]
        point_scores = (diffs * alpha_np).sum(axis=0) / (alpha_np.sum() + 1e-9)
        
        top_errors = np.argsort(-point_scores)[:5]
        top_error_indices = [self.lip_indices[i] for i in top_errors]

        # [PASS 기준도 엄격하게]
        # 80점 이상 (즉, 유사도 0.976 이상)이어야 통과
        res = {
            "label": self.label,
            "similarity": cos_sim,
            "score": score,
            "passed": score >= 80.0, 
            "top_error_landmarks": [int(i) for i in top_error_indices]
        }
        
        if out_report and frames is not None:
            self.save_report_img(frames, alpha, score, out_report)
            
        return res

    def save_report_img(self, frames, alpha, score, path):
        idx = np.argmax(alpha)
        if idx >= len(frames): idx = len(frames)-1
        img = frames[idx].copy()
        
        # 점수에 따라 색상 변경 (80점 기준)
        color = (0, 255, 0) if score >= 80.0 else (0, 0, 255)
        cv2.putText(img, f"Score: {score:.1f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.imwrite(path, img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--refpack", required=True)
    parser.add_argument("--json", action="store_true")
    
    # 이미지 저장 옵션 (기본값 설정)
    parser.add_argument("--out_img", default="result_report.jpg", help="Output report image path")
    
    args = parser.parse_args()
    
    eng = PronunciationEvaluator(args.ckpt, args.refpack)
    
    # 이미지 저장 경로 전달
    result = eng.evaluate(args.input, is_json=args.json, out_report=args.out_img)
    
    print(json.dumps(result, indent=2))