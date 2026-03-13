from __future__ import annotations

import cv2
import logging
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from models import get_resource_path

# ── Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Type Aliases ─────────────────────────────────────────────────────
BGRImage = NDArray[np.uint8]
F32Array = NDArray[np.float32]

# ── YuNet Constants ───────────────────────────────────────────────────
_YUNET_PATH = str(get_resource_path("public/models/face_detection_yunet_2023mar.onnx"))
_INPUT_SIZE = (640, 640)
_SCORE_THRESH = 0.4
_NMS_THRESH = 0.2

_DETECTOR = cv2.FaceDetectorYN.create(
    model=_YUNET_PATH,
    config="",
    input_size=_INPUT_SIZE,
    score_threshold=_SCORE_THRESH,
    nms_threshold=_NMS_THRESH,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU,
)

# ── Model Paths ──────────────────────────────────────────────────────
_PFLD_PATH = str(get_resource_path("public/models/pfld.onnx"))
_DDD_PATH = str(get_resource_path("public/models/ddd_fatigue_v3.onnx"))

# ═════════════════════════════════════════════════════════════════════
# 1. Face Detection
# ═════════════════════════════════════════════════════════════════════


def detect_faces(img: BGRImage) -> F32Array | None:
    h, w = img.shape[:2]
    # 强制将输入尺寸同步给检测器
    _DETECTOR.setInputSize((w, h))
    _, faces = _DETECTOR.detect(img)

    if faces is None:
        # 如果这里打印了，说明 YuNet 确实没看到人脸
        logger.debug(f"YuNet: No faces detected at resolution {w}x{h}")
    else:
        logger.debug(f"YuNet: Detected {len(faces)} face(s)")
    return faces


# ═════════════════════════════════════════════════════════════════════
# 2. Fatigue Monitor (DDD-Centric)
# ═════════════════════════════════════════════════════════════════════


class FatigueMonitor:
    def __init__(self, pfld_path: str, ddd_path: str):
        self.ddd_session = ort.InferenceSession(
            ddd_path, providers=["CPUExecutionProvider"]
        )
        self.pfld_session = ort.InferenceSession(
            pfld_path, providers=["CPUExecutionProvider"]
        )
        self.pfld_input_name = self.pfld_session.get_inputs()[0].name

        self.SLEEP_FRAMES = 5
        self.sleep_counter = 0
        self._last_status = "Normal"

    def _preprocess_ddd(self, face_roi: BGRImage) -> F32Array:
        """为 DDD 模型准备输入: 224x224, ImageNet 归一化, float32"""
        face_resized = cv2.resize(face_roi, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # 归一化并显式转换成 float32 解决之前的 InvalidArgument 错误
        input_data = face_rgb.astype(np.float32) / 255.0
        input_data = (input_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return input_data.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def analyze(
        self, img: BGRImage, face_row: NDArray, need_landmarks: bool = False
    ) -> dict:
        h_img, w_img = img.shape[:2]
        # YuNet row: [x, y, w, h, ...]
        x, y, w, h = map(int, face_row[0:4])

        # ── 1. 提取整脸 ROI 并扩边 ──
        pad_w, pad_h = int(w * 0.05), int(h * 0.05)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(w_img, x + w + pad_w), min(h_img, y + h + pad_h)

        face_roi = img[y1:y2, x1:x2]

        if face_roi.size == 0:
            logger.warning(f"Empty ROI for face at [{x}, {y}]")
            return {}

        # 2. 增强对比度 (可选，如果环境光线暗，开启此逻辑能显著提升置信度)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        face_roi = cv2.merge((cl, a, b))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_LAB2BGR)

        # ── 3. DDD 模型推理 ──
        ddd_blob = self._preprocess_ddd(face_roi)
        ddd_outputs = self.ddd_session.run(None, {"input": ddd_blob})
        label_id = np.argmax(ddd_outputs[0])  # 0: Normal, 1: Drowsy

        # 计算 Softmax 置信度
        logits = ddd_outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        drowsy_prob = probs[1]
        label_id = np.argmax(probs)

        logging.debug(f"DDD: label_id: {label_id} (drowsy_prob: {drowsy_prob:.2f})")

        # ── 3.疲劳判定逻辑 ──
        status = "Normal"
        if label_id == 1 or drowsy_prob > 0.5:
            status = "Drowsy Alert"

        if label_id == 1:
            self.sleep_counter += 1
            if self.sleep_counter >= self.SLEEP_FRAMES:
                status = "Drowsy Alert"
        else:
            self.sleep_counter = 0

        if status != self._last_status:
            logger.info(f"Status Change: {self._last_status} -> {status}")
            self._last_status = status

        return {
            "status": status,
            "bbox": [x, y, w, h],
            "confidence": float(drowsy_prob),
            "is_complete": True,
        }


# ── 初始化单例 ──
_monitor = FatigueMonitor(_PFLD_PATH, _DDD_PATH)


def process_and_analyze(img: BGRImage, show_box: bool = True) -> BGRImage:
    h, w = img.shape[:2]
    _DETECTOR.setInputSize((w, h))
    _, faces = _DETECTOR.detect(img)
    result = img.copy()

    if faces is None:
        return result

    for face in faces:
        analysis = _monitor.analyze(img, face)
        if not analysis or "bbox" not in analysis:
            continue

        if show_box:
            x, y, wb, hb = analysis["bbox"]
            color = (0, 0, 255) if "Alert" in analysis["status"] else (0, 255, 0)
            cv2.rectangle(result, (x, y), (x + wb, y + hb), color, 2)
            cv2.putText(
                result,
                analysis["status"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    return result
