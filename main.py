import argparse
import base64
import io
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ExifTags

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:  # pragma: no cover
    pytesseract = None
    TesseractNotFoundError = Exception

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

try:
    from pyzbar import pyzbar
except Exception:  # pragma: no cover
    pyzbar = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    from exif import Image as ExifImage
except Exception:  # pragma: no cover
    ExifImage = None

try:
    from skimage.measure import shannon_entropy
except Exception:  # pragma: no cover
    shannon_entropy = None

try:
    from colorama import Fore, Style, init as colorama_init
except Exception:  # pragma: no cover
    Fore = Style = None
    colorama_init = None

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except Exception:  # pragma: no cover
    Environment = FileSystemLoader = select_autoescape = None

try:
    from google.cloud import vision
except Exception:  # pragma: no cover
    vision = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
KEYWORDS = {"confidential", "password", "login", "seed", "recovery"}
DEFAULT_BIP39 = {
    "abandon",
    "ability",
    "able",
    "about",
    "above",
    "absent",
    "absorb",
    "abstract",
    "absurd",
    "abuse",
    "access",
    "accident",
    "account",
    "accuse",
    "achieve",
    "acid",
    "acoustic",
    "acquire",
    "across",
    "act",
    "action",
    "actor",
    "actress",
    "actual",
    "adapt",
    "add",
    "addict",
    "address",
    "adjust",
    "admit",
    "adult",
    "advance",
    "advice",
    "aerobic",
    "affair",
    "afford",
    "afraid",
    "again",
    "age",
    "agent",
    "agree",
    "ahead",
    "aim",
    "air",
    "airport",
    "aisle",
    "alarm",
    "album",
    "alcohol",
    "alert",
    "alien",
    "all",
    "alley",
    "allow",
    "almost",
    "alone",
    "alpha",
    "already",
    "also",
    "alter",
    "always",
    "amateur",
    "amazing",
    "among",
    "amount",
    "amused",
    "analyst",
    "anchor",
    "ancient",
    "anger",
    "angle",
    "angry",
    "animal",
    "ankle",
    "announce",
    "annual",
    "another",
    "answer",
    "antenna",
    "antique",
    "anxiety",
    "any",
    "apart",
    "apology",
    "appear",
    "apple",
    "approve",
    "april",
    "arch",
    "arctic",
    "area",
    "arena",
    "argue",
    "arm",
    "armed",
    "armor",
    "army",
    "around",
    "arrange",
    "arrest",
    "arrive",
    "arrow",
    "art",
    "artefact",
    "artist",
    "artwork",
    "ask",
    "aspect",
    "assault",
    "asset",
    "assist",
    "assume",
    "asthma",
    "athlete",
    "atom",
    "attack",
    "attend",
    "attitude",
    "attract",
    "auction",
    "audit",
    "august",
    "aunt",
    "author",
    "auto",
    "autumn",
    "average",
    "avocado",
    "avoid",
    "awake",
    "aware",
    "away",
    "awesome",
    "awful",
    "awkward",
    "axis",
    "baby",
    "bachelor",
    "bacon",
    "badge",
    "bag",
    "balance",
    "balcony",
    "ball",
    "bamboo",
    "banana",
    "banner",
    "bar",
    "barely",
    "bargain",
    "barrel",
    "base",
    "basic",
    "basket",
    "battle",
    "beach",
    "bean",
    "beauty",
    "because",
    "become",
    "beef",
    "before",
    "begin",
    "behave",
    "behind",
    "believe",
    "below",
    "belt",
    "bench",
    "benefit",
    "best",
    "betray",
    "better",
    "between",
    "beyond",
    "bicycle",
    "bid",
    "bike",
    "bind",
    "biology",
    "bird",
    "birth",
    "bitter",
    "black",
    "blade",
    "blame",
    "blanket",
    "blast",
    "bleak",
    "bless",
    "blind",
    "blood",
    "blossom",
    "blouse",
    "blue",
    "blur",
    "blush",
    "board",
    "boat",
    "body",
    "boil",
    "bomb",
    "bone",
    "bonus",
    "book",
    "boost",
    "border",
    "boring",
    "borrow",
    "boss",
    "bottom",
    "bounce",
    "box",
    "boy",
    "bracket",
    "brain",
    "brand",
    "brass",
    "brave",
    "bread",
    "breeze",
    "brick",
    "bridge",
    "brief",
    "bright",
    "bring",
    "brisk",
    "broccoli",
    "broken",
    "bronze",
    "broom",
    "brother",
    "brown",
    "brush",
    "bubble",
    "buddy",
    "budget",
    "buffalo",
    "build",
    "bulb",
    "bulk",
    "bullet",
    "bundle",
    "bunker",
    "burden",
    "burger",
    "burst",
    "bus",
    "business",
    "busy",
    "butter",
    "buyer",
    "buzz",
}


SEVERITY_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
YOLO_TARGETS = {"cell phone", "laptop", "credit card", "handbag", "backpack"}
VISION_KEYWORDS = {
    "credit card": "High",
    "card": "Medium",
    "laptop": "Medium",
    "computer": "Medium",
    "cell phone": "Medium",
    "mobile phone": "Medium",
    "smartphone": "Medium",
    "gun": "High",
    "firearm": "High",
    "rifle": "High",
    "handgun": "High",
    "weapon": "High",
}


@dataclass
class Finding:
    module: str
    severity: str
    summary: str
    details: Optional[str] = None


@dataclass
class EvidenceItem:
    path: str
    rel_path: str
    findings: List[Finding] = field(default_factory=list)
    ocr_snippet: Optional[str] = None
    qr_data: Optional[str] = None
    vision_objects: List[str] = field(default_factory=list)
    vision_labels: List[str] = field(default_factory=list)
    geoint_link: Optional[str] = None
    entropy: Optional[float] = None
    stego_payload: Optional[str] = None
    stego_message: Optional[str] = None
    stego_reason: Optional[str] = None
    stego_artifact_path: Optional[str] = None
    stego_artifact_thumbnail: Optional[str] = None
    stego_ocr: Optional[str] = None
    yolo_labels: List[str] = field(default_factory=list)

    @property
    def top_severity(self) -> str:
        if not self.findings:
            return "Low"
        return max(self.findings, key=lambda f: SEVERITY_ORDER[f.severity]).severity


def normalize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def load_bip39(path: Optional[str]) -> set:
    if not path:
        return set(DEFAULT_BIP39)
    words = set()
    try:
        if path.startswith("http://") or path.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(path, timeout=10) as response:
                for line in response.read().decode("utf-8").splitlines():
                    word = line.strip().lower()
                    if word:
                        words.add(word)
        else:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
    except FileNotFoundError:
        print(f"[WARN] BIP39 list not found at {path}. Using default mini-list.")
        return set(DEFAULT_BIP39)
    except Exception:
        print(f"[WARN] Failed to load BIP39 list from {path}. Using default mini-list.")
        return set(DEFAULT_BIP39)
    return words


def load_image_cv2(path: str):
    if not cv2:
        return None
    return cv2.imread(path)


def configure_tesseract():
    if not pytesseract:
        return
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path


def load_env():
    # Load .env from repo root (same dir as this file) or cwd.
    repo_env = os.path.join(os.path.dirname(__file__), ".env")
    env_path = repo_env if os.path.exists(repo_env) else ".env"
    if load_dotenv:
        load_dotenv(env_path)
    # Lightweight fallback parser to handle unquoted values with spaces.
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value.strip().strip('"')
        except Exception:
            pass


def resolve_credentials_path() -> str:
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return os.path.expandvars(os.path.expanduser(raw))


def ensure_inline_credentials():
    inline_json = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON") or "").strip()
    if not inline_json:
        return
    if inline_json.startswith('"') and inline_json.endswith('"'):
        inline_json = inline_json[1:-1]
    # Allow base64-encoded JSON to avoid escaping issues in .env.
    if inline_json.lower().startswith("base64:"):
        try:
            inline_json = base64.b64decode(inline_json.split(":", 1)[1]).decode("utf-8")
        except Exception:
            return
    if not inline_json.lstrip().startswith("{"):
        return
    creds_path = os.path.join(os.path.dirname(__file__), ".vision_credentials.json")
    try:
        with open(creds_path, "w", encoding="utf-8") as handle:
            handle.write(inline_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    except Exception:
        return


_easyocr_reader = None


def get_easyocr_reader():
    global _easyocr_reader
    if not easyocr:
        return None
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            _easyocr_reader = None
    return _easyocr_reader


def run_ocr(image_bgr, bip39_words: set, enabled: bool) -> Tuple[Optional[str], List[Finding]]:
    findings: List[Finding] = []
    if not enabled or image_bgr is None:
        return None, findings
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        text = ""
        if pytesseract:
            text = pytesseract.image_to_string(gray) or ""
        if not text and easyocr:
            reader = get_easyocr_reader()
            if reader:
                result = reader.readtext(gray)
                text = " ".join([r[1] for r in result])
    except TesseractNotFoundError:
        text = ""
    except Exception:
        text = ""
    if not text:
        return None, findings
    words = normalize_words(text)
    bip_hits = [w for w in words if w in bip39_words]
    keyword_hits = [w for w in words if w in KEYWORDS]
    if len(set(bip_hits)) >= 5 or keyword_hits:
        summary = "OCR match"
        details = None
        if keyword_hits:
            details = f"Keyword(s): {', '.join(sorted(set(keyword_hits)))}"
        elif bip_hits:
            details = f"BIP39 words: {', '.join(sorted(set(bip_hits))[:8])}"
        findings.append(
            Finding(
                module="OCR",
                severity="Critical",
                summary=summary,
                details=details,
            )
        )
    snippet = text.strip().replace("\n", " ")
    return (snippet[:200] if snippet else None), findings


def looks_like_crypto(data: str) -> bool:
    if data.startswith("bc1") or data.startswith("1") or data.startswith("3"):
        return True
    if re.fullmatch(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}", data):
        return True
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", data):
        return True
    return False


def looks_like_url(data: str) -> bool:
    return bool(re.match(r"^(https?://|www\.)", data.lower()))


def run_qr(image_bgr) -> Tuple[Optional[str], List[Finding]]:
    findings: List[Finding] = []
    payloads: List[str] = []
    if image_bgr is None:
        return None, findings
    if pyzbar:
        decoded = pyzbar.decode(image_bgr)
        for entry in decoded:
            data = entry.data.decode("utf-8", errors="ignore")
            if data:
                payloads.append(data)
    if not payloads and cv2:
        try:
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(image_bgr)
            if data:
                payloads.append(data)
        except Exception:
            pass
    if not payloads:
        return None, findings
    payload = payloads[0]
    severity = "High" if (looks_like_url(payload) or looks_like_crypto(payload)) else "Medium"
    details = "URL" if looks_like_url(payload) else "Crypto/Other" if looks_like_crypto(payload) else "QR data"
    findings.append(
        Finding(
            module="QR",
            severity=severity,
            summary="QR code found",
            details=f"{details}: {payload[:200]}",
        )
    )
    if len(payloads) > 1:
        findings.append(
            Finding(
                module="QR",
                severity="Low",
                summary="Multiple QR payloads",
                details=f"Count: {len(payloads)}",
            )
        )
    return payload, findings


def load_yolo(model_path: str):
    if not YOLO:
        return None
    try:
        return YOLO(model_path)
    except Exception:
        return None


def run_yolo(model, image_path: str, conf: float, imgsz: int) -> Tuple[List[str], List[Finding]]:
    findings: List[Finding] = []
    labels: List[str] = []
    if not model:
        return labels, findings
    try:
        results = model(image_path, verbose=False, conf=conf, imgsz=imgsz)
    except Exception:
        return labels, findings
    for result in results:
        if not result.names:
            continue
        for cls_id in result.boxes.cls.tolist() if result.boxes is not None else []:
            name = result.names.get(int(cls_id))
            if not name:
                continue
            labels.append(name)
            if name in YOLO_TARGETS:
                findings.append(
                    Finding(
                        module="YOLO",
                        severity="Medium",
                        summary="Object detected",
                        details=name,
                    )
                )
    return sorted(set(labels)), findings


_vision_client = None


def get_vision_client():
    global _vision_client
    if not vision:
        return None
    if _vision_client is None:
        try:
            _vision_client = vision.ImageAnnotatorClient()
        except Exception:
            _vision_client = None
    return _vision_client


def classify_vision_label(name: str) -> Optional[str]:
    name_l = name.lower()
    for keyword, severity in VISION_KEYWORDS.items():
        if keyword in name_l:
            return severity
    return None


def run_vision(
    path: str,
    min_score: float = 0.4,
    debug: bool = False,
) -> Tuple[Optional[str], List[str], List[str], List[Finding]]:
    findings: List[Finding] = []
    objects: List[str] = []
    labels: List[str] = []
    client = get_vision_client()
    if not client:
        return None, objects, labels, findings
    try:
        with open(path, "rb") as handle:
            content = handle.read()
    except Exception:
        return None, objects, labels, findings
    image = vision.Image(content=content)
    try:
        response = client.annotate_image(
            {
                "image": image,
                "features": [
                    {"type_": vision.Feature.Type.OBJECT_LOCALIZATION},
                    {"type_": vision.Feature.Type.LABEL_DETECTION},
                    {"type_": vision.Feature.Type.TEXT_DETECTION},
                    {"type_": vision.Feature.Type.LOGO_DETECTION},
                ],
            }
        )
    except Exception as e:
        if debug:
            print_status("[WARN]", f"Vision API exception: {e}")
        return None, objects, labels, findings
    if response.error.message:
        if debug:
            print_status("[WARN]", f"Vision API error: {response.error.message}")
        return None, objects, labels, findings

    # Extract text (can contain QR code content if scanned)
    payload = None
    text_annotations = getattr(response, "text_annotations", None) or []
    if text_annotations:
        full_text = text_annotations[0].description if text_annotations else ""
        if full_text:
            # Check if text looks like QR/barcode data
            if looks_like_url(full_text) or looks_like_crypto(full_text):
                payload = full_text.strip()[:500]
                severity = "High" if (looks_like_url(payload) or looks_like_crypto(payload)) else "Medium"
                details = "URL" if looks_like_url(payload) else "Crypto/Other" if looks_like_crypto(payload) else "Text data"
                findings.append(
                    Finding(
                        module="VISION",
                        severity=severity,
                        summary="Text/QR content found",
                        details=f"{details}: {payload[:200]}",
                    )
                )

    # Extract logos
    for logo in getattr(response, "logo_annotations", None) or []:
        if logo.score < min_score:
            continue
        name = logo.description or ""
        if name:
            labels.append(f"Logo: {name} ({logo.score:.2f})")

    for obj in response.localized_object_annotations or []:
        if obj.score < min_score:
            continue
        name = obj.name or ""
        if name:
            objects.append(name)
            severity = classify_vision_label(name)
            if severity:
                findings.append(
                    Finding(
                        module="VISION",
                        severity=severity,
                        summary="Object detected",
                        details=f"{name} ({obj.score:.2f})",
                    )
                )

    label_items = response.label_annotations or []
    if debug and not label_items and not response.localized_object_annotations:
        print_status("[WARN]", "Vision API returned no labels/objects.")
    for label in label_items:
        if label.score < min_score:
            continue
        name = label.description or ""
        if not name:
            continue
        labels.append(f"{name} ({label.score:.2f})")
        severity = classify_vision_label(name)
        if severity:
            findings.append(
                Finding(
                    module="VISION",
                    severity=severity,
                    summary="Label detected",
                    details=f"{name} ({label.score:.2f})",
                )
            )
            if name not in objects:
                objects.append(name)

    return payload, sorted(set(objects)), labels, findings


def dms_to_decimal(dms: Tuple, ref: str) -> Optional[float]:
    try:
        degrees = float(dms[0].numerator) / float(dms[0].denominator)
        minutes = float(dms[1].numerator) / float(dms[1].denominator)
        seconds = float(dms[2].numerator) / float(dms[2].denominator)
    except Exception:
        try:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
        except Exception:
            return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in {"S", "W"}:
        decimal *= -1
    return round(decimal, 6)


def extract_gps_from_exif(path: str) -> Optional[str]:
    if ExifImage:
        try:
            with open(path, "rb") as handle:
                exif_img = ExifImage(handle)
            if not exif_img.has_exif:
                return None
            lat = getattr(exif_img, "gps_latitude", None)
            lat_ref = getattr(exif_img, "gps_latitude_ref", None)
            lon = getattr(exif_img, "gps_longitude", None)
            lon_ref = getattr(exif_img, "gps_longitude_ref", None)
            if lat and lon and lat_ref and lon_ref:
                lat_dec = dms_to_decimal(lat, lat_ref)
                lon_dec = dms_to_decimal(lon, lon_ref)
                if lat_dec is None or lon_dec is None:
                    return None
                return f"https://maps.google.com/?q={lat_dec},{lon_dec}"
        except Exception:
            return None
    try:
        image = Image.open(path)
        exif_data = image._getexif() or {}
    except Exception:
        return None
    gps_info = None
    for tag, value in exif_data.items():
        if ExifTags.TAGS.get(tag) == "GPSInfo":
            gps_info = value
            break
    if not gps_info:
        return None
    def gps_value(key):
        return gps_info.get(key)
    lat = gps_value(2)
    lat_ref = gps_value(1)
    lon = gps_value(4)
    lon_ref = gps_value(3)
    if not (lat and lon and lat_ref and lon_ref):
        return None
    lat_dec = dms_to_decimal(lat, lat_ref)
    lon_dec = dms_to_decimal(lon, lon_ref)
    if lat_dec is None or lon_dec is None:
        return None
    return f"https://maps.google.com/?q={lat_dec},{lon_dec}"


def run_geoint(path: str) -> Tuple[Optional[str], List[Finding]]:
    findings: List[Finding] = []
    link = extract_gps_from_exif(path)
    if link:
        findings.append(
            Finding(
                module="GEOINT",
                severity="Medium",
                summary="Location data found",
                details=link,
            )
        )
    return link, findings


def trailing_bytes_count(path: str) -> int:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except Exception:
        return 0
    lower = path.lower()
    if lower.endswith(".png"):
        signature = b"\x00\x00\x00\x00IEND\xaeB`\x82"
        index = data.rfind(signature)
        if index != -1:
            end = index + len(signature)
            return max(0, len(data) - end)
    if lower.endswith((".jpg", ".jpeg")):
        eoi = b"\xff\xd9"
        index = data.rfind(eoi)
        if index != -1:
            end = index + len(eoi)
            return max(0, len(data) - end)
    if lower.endswith(".bmp") and len(data) >= 6:
        declared = int.from_bytes(data[2:6], byteorder="little", signed=False)
        if declared and len(data) > declared:
            return len(data) - declared
    return 0


def lsb_chi_square(gray) -> Optional[float]:
    try:
        lsb = (gray & 1).ravel()
        zeros = int((lsb == 0).sum())
        ones = int((lsb == 1).sum())
        total = zeros + ones
        if total == 0:
            return None
        expected = total / 2.0
        chi2 = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
        return float(chi2)
    except Exception:
        return None


def high_frequency_energy(gray) -> Optional[float]:
    if not cv2:
        return None
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception:
        return None


def run_entropy(image_bgr, path: str) -> Tuple[Optional[float], List[Finding]]:
    findings: List[Finding] = []
    trailing = trailing_bytes_count(path)
    if trailing > 1024:
        findings.append(
            Finding(
                module="STEGO",
                severity="Medium",
                summary="Trailing data detected",
                details=f"Trailing bytes: {trailing}",
            )
        )
    if image_bgr is None or not shannon_entropy:
        return None, findings
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    entropy = float(shannon_entropy(gray))
    chi2 = lsb_chi_square(gray)
    hf_energy = high_frequency_energy(gray)
    high_entropy = entropy > 7.6
    suspect_lsb = chi2 is not None and chi2 < 1.2
    high_noise = hf_energy is not None and hf_energy > 200.0
    if high_entropy or (entropy > 7.2 and (suspect_lsb or high_noise)):
        details_parts = [f"Entropy: {entropy:.2f}"]
        if chi2 is not None:
            details_parts.append(f"LSB chi2: {chi2:.2f}")
        if hf_energy is not None:
            details_parts.append(f"HF energy: {hf_energy:.1f}")
        findings.append(
            Finding(
                module="STEGO",
                severity="Medium",
                summary="Suspected steganography / high noise",
                details=" | ".join(details_parts),
            )
        )
    return entropy, findings


def decode_stylesuxx_steganography(image_path: str) -> Optional[str]:
    """Decode steganography using stylesuxx.github.io algorithm.
    This matches the exact algorithm from the website:
    - Even RGB values = 0 bit, Odd RGB values = 1 bit
    - 3 bits per pixel (R, G, B channels)
    - 8 bits per character
    - Matches the exact JavaScript bit order and processing
    """
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        pixels = list(image.get_flattened_data())
        
        # Extract binary message from RGB values (exact JavaScript algorithm)
        binary_message = ""
        for pixel in pixels:
            for channel_idx in range(3):  # R=0, G=1, B=2
                channel_value = pixel[channel_idx]
                # Even = 0, Odd = 1 (matching website algorithm)
                if channel_value % 2 != 0:
                    binary_message += "1"
                else:
                    binary_message += "0"
        
        # Convert binary to ASCII characters (8 bits each, exact JavaScript method)
        decoded_text = ""
        consecutive_nulls = 0
        
        for i in range(0, len(binary_message), 8):
            if i + 8 > len(binary_message):
                break
            
            byte_str = binary_message[i:i+8]
            if len(byte_str) == 8:
                # Match JavaScript: c <<= 1; c |= parseInt(binaryMessage[i + j]);
                char_code = 0
                for j in range(8):
                    char_code <<= 1
                    char_code |= int(byte_str[j])
                
                # Handle different character types
                if char_code == 0:
                    consecutive_nulls += 1
                    if consecutive_nulls >= 3:  # Multiple nulls = end of message
                        break
                elif 32 <= char_code <= 126:  # Printable ASCII
                    decoded_text += chr(char_code)
                    consecutive_nulls = 0
                elif char_code in [9, 10, 13]:  # Tab, newline, carriage return
                    decoded_text += chr(char_code)
                    consecutive_nulls = 0
                else:
                    # Non-printable character
                    if len(decoded_text) >= 8:  # If we have substantial text, stop
                        break
                    elif len(decoded_text) == 0 and i > 800:  # No text found after many attempts
                        break
        
        # Clean up the decoded text
        decoded_text = decoded_text.strip()
        
        # More stringent validation for real messages
        if len(decoded_text) >= 5:
            # Check for reasonable character distribution
            alpha_count = sum(c.isalpha() for c in decoded_text)
            printable_count = sum(32 <= ord(c) <= 126 for c in decoded_text)
            
            # Must have at least some letters and mostly printable characters
            if alpha_count >= 3 and printable_count >= len(decoded_text) * 0.8:
                # Additional check: shouldn't be mostly repetitive
                unique_chars = len(set(decoded_text.lower()))
                if unique_chars >= min(4, len(decoded_text) // 3):
                    return decoded_text
            
    except Exception:
        pass
    
    return None


def extract_lsb_bytes(raw_bytes: bytes, bit: int = 0, max_output: int = 200000) -> bytes:
    if not raw_bytes:
        return b""
    max_input = min(len(raw_bytes), max_output * 8)
    out = bytearray()
    current = 0
    bit_count = 0
    for b in raw_bytes[:max_input]:
        current = (current << 1) | ((b >> bit) & 1)
        bit_count += 1
        if bit_count == 8:
            out.append(current)
            current = 0
            bit_count = 0
    return bytes(out)


def longest_printable_ascii(data: bytes, min_len: int = 24) -> Optional[str]:
    if not data:
        return None
    best = ""
    current = []
    for b in data:
        if b in (9, 10, 13) or 32 <= b <= 126:
            current.append(chr(b))
        else:
            if len(current) > len(best):
                best = "".join(current)
            current = []
    if len(current) > len(best):
        best = "".join(current)
    if len(best) >= min_len:
        return best.strip()
    return None


def detect_magic(data: bytes) -> Optional[Tuple[str, int]]:
    signatures = [
        (b"\x89PNG\r\n\x1a\n", "PNG"),
        (b"\xff\xd8\xff", "JPEG"),
        (b"GIF87a", "GIF"),
        (b"GIF89a", "GIF"),
        (b"PK\x03\x04", "ZIP"),
        (b"PK\x05\x06", "ZIP"),
        (b"%PDF", "PDF"),
        (b"BM", "BMP"),
        (b"RIFF", "RIFF"),
    ]
    for sig, name in signatures:
        idx = data.find(sig)
        if idx != -1:
            return name, idx
    return None


def find_base64_blob(text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z0-9+/=]{80,}", text)
    if match:
        return match.group(0)
    return None


def decode_exif_text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            if len(value) > 1 and value[1] == 0:
                return value.decode("utf-16le", errors="ignore").strip("\x00")
            return value.decode("utf-8", errors="ignore").strip("\x00")
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        try:
            as_bytes = bytes(value)
            return decode_exif_text_value(as_bytes)
        except Exception:
            return None
    try:
        text = str(value)
        return text.strip() if text else None
    except Exception:
        return None


def extract_exif_text(path: str) -> Optional[str]:
    keys = {
        "user_comment",
        "image_description",
        "xp_comment",
        "xp_title",
        "xp_subject",
        "xp_keywords",
        "comment",
    }
    candidates: List[str] = []
    if ExifImage:
        try:
            with open(path, "rb") as handle:
                exif_img = ExifImage(handle)
            if exif_img.has_exif:
                for key in keys:
                    if hasattr(exif_img, key):
                        value = getattr(exif_img, key, None)
                        text = decode_exif_text_value(value)
                        if text:
                            candidates.append(text)
        except Exception:
            pass
    try:
        image = Image.open(path)
        exif_data = image.getexif() or {}
        for tag, value in exif_data.items():
            name = ExifTags.TAGS.get(tag, "").lower()
            if name in keys:
                text = decode_exif_text_value(value)
                if text:
                    candidates.append(text)
    except Exception:
        pass
    if not candidates:
        return None
    combined = " | ".join(sorted(set(candidates)))
    return combined[:300] if combined else None


def scan_file_strings(path: str, min_len: int = 32, max_bytes: int = 3_000_000) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            data = handle.read(max_bytes)
    except Exception:
        return None
    text = longest_printable_ascii(data, min_len=min_len)
    if text:
        return text[:300]
    try:
        text16 = data.decode("utf-16le", errors="ignore")
        match = re.search(r"[A-Za-z0-9\s\-_,.:/]{%d,}" % min_len, text16)
        if match:
            return match.group(0)[:300]
    except Exception:
        pass
    return None


def extract_embedded_file(data: bytes, name: str, offset: int) -> Tuple[Optional[bytes], Optional[str]]:
    if offset < 0 or offset >= len(data):
        return None, None
    blob = data[offset:]
    if name == "BMP" and len(blob) >= 6:
        size = int.from_bytes(blob[2:6], byteorder="little", signed=False)
        if 0 < size <= len(blob):
            return blob[:size], "bmp"
        if size == 0 and len(blob) >= 1024:
            fallback = blob[: min(len(blob), 5_000_000)]
            return fallback, "bmp"
    if name == "PNG":
        signature = b"\x00\x00\x00\x00IEND\xaeB`\x82"
        idx = blob.find(signature)
        if idx != -1:
            end = idx + len(signature)
            return blob[:end], "png"
    if name == "JPEG":
        idx = blob.find(b"\xff\xd9")
        if idx != -1:
            end = idx + 2
            return blob[:end], "jpg"
    if name == "GIF":
        idx = blob.find(b"\x3b")
        if idx != -1:
            end = idx + 1
            return blob[:end], "gif"
    if name == "PDF":
        idx = blob.find(b"%%EOF")
        if idx != -1:
            end = idx + 5
            return blob[:end], "pdf"
    return None, None


def analyze_lsb_payload(data: bytes) -> Tuple[Optional[str], Optional[str], int, Optional[bytes], Optional[str]]:
    magic = detect_magic(data)
    if magic:
        name, offset = magic
        preview = data[offset : offset + 64].hex()
        details = f"{name} signature at offset {offset}. Bytes: {preview}"
        extracted, ext = extract_embedded_file(data, name, offset)
        return f"{name} embedded data", details, 3, extracted, ext

    text = longest_printable_ascii(data)
    if text:
        blob = find_base64_blob(text)
        if blob:
            try:
                decoded = base64.b64decode(blob, validate=True)
            except Exception:
                decoded = b""
            if decoded:
                decoded_text = longest_printable_ascii(decoded, min_len=16)
                if decoded_text:
                    preview = decoded_text[:300]
                    return "Base64 text decoded", preview, 3, None, None
                magic2 = detect_magic(decoded)
                if magic2:
                    name2, offset2 = magic2
                    preview = decoded[offset2 : offset2 + 64].hex()
                    details = f"Decoded {name2} signature at offset {offset2}. Bytes: {preview}"
                    return "Base64 file decoded", details, 3, None, None
        return "LSB text decoded", text[:300], 2, None, None
    return None, None, 0, None, None


def extract_trailing_payload(path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except Exception:
        return None, None
    lower = path.lower()
    trailing = 0
    start = 0
    if lower.endswith(".png"):
        signature = b"\x00\x00\x00\x00IEND\xaeB`\x82"
        idx = data.rfind(signature)
        if idx != -1:
            start = idx + len(signature)
            trailing = len(data) - start
    elif lower.endswith((".jpg", ".jpeg")):
        eoi = b"\xff\xd9"
        idx = data.rfind(eoi)
        if idx != -1:
            start = idx + len(eoi)
            trailing = len(data) - start
    elif lower.endswith(".bmp") and len(data) >= 6:
        declared = int.from_bytes(data[2:6], byteorder="little", signed=False)
        if declared and len(data) > declared:
            start = declared
            trailing = len(data) - start
    if trailing <= 0:
        return None, None
    blob = data[start:]
    magic = detect_magic(blob)
    if magic:
        name, offset = magic
        preview = blob[offset : offset + 64].hex()
        return "Trailing embedded data", f"{name} signature at offset {offset}. Bytes: {preview}"
    text = longest_printable_ascii(blob, min_len=24)
    if text:
        return "Trailing text decoded", text[:300]
    return None, None


def run_stego_decode(path: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[Finding], Optional[str]]:
    findings: List[Finding] = []
    payload_preview: Optional[str] = None
    human_message: Optional[str] = None
    critical_reason: Optional[str] = None
    artifact_path: Optional[str] = None
    
    # Try stylesuxx steganography algorithm first (most likely match)
    stylesuxx_message = decode_stylesuxx_steganography(path)
    if stylesuxx_message:
        human_message = stylesuxx_message
        critical_reason = "Hidden message decoded using stylesuxx steganography algorithm"
        findings.append(
            Finding(
                module="STEGO",
                severity="Critical",
                summary="Stylesuxx steganography decoded",
                details=f"Message: {stylesuxx_message[:200]}",
            )
        )
        # Return early if we found the message using the specific algorithm
        return None, human_message, critical_reason, findings, artifact_path
    
    try:
        image = Image.open(path).convert("RGB")
        raw = image.tobytes()
    except Exception:
        raw = b""

    candidates = []
    if raw:
        candidates.extend(
            [
                ("rgb", raw),
                ("r", raw[0::3]),
                ("g", raw[1::3]),
                ("b", raw[2::3]),
            ]
        )
    best_score = 0
    best_summary = None
    best_details = None
    best_mode = None
    best_blob = None
    best_ext = None
    for mode, data in candidates:
        for bit in range(0, 3):
            lsb_bytes = extract_lsb_bytes(data, bit=bit)
            summary, details, score, extracted, ext = analyze_lsb_payload(lsb_bytes)
            if score > best_score:
                best_score = score
                best_summary = summary
                best_details = details
                best_mode = f"{mode}/bit{bit}"
                best_blob = extracted
                best_ext = ext
                # If we found readable text, prioritize it as the human message
                if score == 2 and "LSB text decoded" in (summary or ""):
                    human_message = details[:300]
                    critical_reason = f"Hidden text decoded from {mode} channel bit {bit}"

    trail_summary, trail_details = extract_trailing_payload(path)
    if trail_summary:
        findings.append(
            Finding(
                module="STEGO",
                severity="Critical",
                summary=trail_summary,
                details=trail_details,
            )
        )
        if trail_details and "text decoded" in trail_summary:
            human_message = trail_details[:300]
            critical_reason = "Hidden text found in trailing file bytes"
        elif trail_details:
            payload_preview = trail_details[:300]
            critical_reason = "Suspicious trailing data detected"

    if best_summary and best_details:
        saved_path = None
        if best_blob and best_ext:
            base = os.path.splitext(os.path.basename(path))[0]
            safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("_")
            out_dir = os.path.join(os.path.dirname(path), "_stego_extract")
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{safe_base}_{best_mode.replace('/', '_')}.{best_ext}"
            saved_path = os.path.join(out_dir, out_name)
            try:
                with open(saved_path, "wb") as handle:
                    handle.write(best_blob)
            except Exception:
                saved_path = None
        if saved_path:
            artifact_path = saved_path
        findings.append(
            Finding(
                module="STEGO",
                severity="Critical",
                summary=best_summary,
                details=f"Mode: {best_mode} | {best_details}" + (f" | Saved: {saved_path}" if saved_path else ""),
            )
        )
        # Don't overwrite readable text payload with technical details
        if not payload_preview or (best_summary == "LSB text decoded"):
            payload_preview = best_details[:300]

    exif_text = extract_exif_text(path)
    if exif_text:
        findings.append(
            Finding(
                module="STEGO",
                severity="Critical",
                summary="Metadata text decoded",
                details=exif_text,
            )
        )
        if not human_message:
            human_message = exif_text[:300]
            critical_reason = "Hidden text found in image metadata (EXIF)"

    file_text = scan_file_strings(path)
    if file_text:
        findings.append(
            Finding(
                module="STEGO",
                severity="Critical",
                summary="Hidden text found in file bytes",
                details=file_text,
            )
        )
        if not human_message:
            human_message = file_text[:300]
            critical_reason = "Hidden text embedded in raw file data"

    # Set default reason if we found something but no text
    if findings and not critical_reason:
        if best_summary:
            critical_reason = f"Embedded {best_summary.lower()} detected using {best_mode} extraction"
        else:
            critical_reason = "Suspicious steganographic patterns detected"
    
    return payload_preview, human_message, critical_reason, findings, artifact_path


def image_to_thumbnail(path: str, max_size: int = 240) -> Optional[str]:
    try:
        image = Image.open(path)
        image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def ocr_image_path(path: str) -> Optional[str]:
    if not cv2:
        return None
    try:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        text = ""
        
        # Try Tesseract first
        if pytesseract:
            try:
                text = pytesseract.image_to_string(gray) or ""
            except Exception:
                text = ""
        
        # If no text from Tesseract, try EasyOCR
        if not text and easyocr:
            try:
                reader = get_easyocr_reader()
                if reader:
                    result = reader.readtext(gray)
                    text = " ".join([r[1] for r in result if r[2] > 0.5])  # Only confident results
            except Exception:
                text = ""
        
        # Also try with different preprocessing
        if not text:
            try:
                # Try with threshold preprocessing
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if pytesseract:
                    text = pytesseract.image_to_string(thresh) or ""
                if not text and easyocr:
                    reader = get_easyocr_reader()
                    if reader:
                        result = reader.readtext(thresh)
                        text = " ".join([r[1] for r in result if r[2] > 0.5])
            except Exception:
                pass
                
    except Exception:
        return None
    snippet = text.strip().replace("\n", " ")
    return snippet[:300] if snippet else None


def print_status(label: str, message: str, color: Optional[str] = None):
    if color and Fore and Style:
        print(f"{color}{label}{Style.RESET_ALL} {message}")
    else:
        print(f"{label} {message}")


def collect_images(root: str) -> List[str]:
    files = []
    for base, _, names in os.walk(root):
        for name in names:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(base, name))
    return sorted(files)


def preflight_checks(yolo_model: str) -> Dict[str, bool]:
    creds_path = resolve_credentials_path()
    status = {
        "ocr": bool(pytesseract or easyocr),
        "qr": bool(pyzbar or cv2),  # OpenCV has built-in QRCodeDetector
        "yolo": bool(YOLO),
        "vision": bool(vision and creds_path and os.path.exists(creds_path)),
        "geoint": True,
        "stego": bool(shannon_entropy),
    }
    status["_vision_path"] = bool(creds_path)
    status["_vision_exists"] = bool(creds_path and os.path.exists(creds_path))
    if status["ocr"]:
        try:
            if pytesseract:
                pytesseract.get_tesseract_version()
        except Exception:
            if not easyocr:
                status["ocr"] = False
    if status["yolo"]:
        try:
            _ = YOLO(yolo_model)
        except Exception:
            status["yolo"] = False
    return status


def print_preflight(status: Dict[str, bool], yolo_model: str):
    print_status("[CHECK]", f"OCR available: {status['ocr']}")
    qr_backend = "pyzbar" if pyzbar else ("OpenCV" if cv2 else "None")
    print_status("[CHECK]", f"QR available: {status['qr']} ({qr_backend})")
    print_status("[CHECK]", f"YOLO available ({yolo_model}): {status['yolo']}")
    print_status("[CHECK]", f"VISION available: {status['vision']}")
    print_status("[CHECK]", f"GEOINT available: {status['geoint']}")
    print_status("[CHECK]", f"STEGO available: {status['stego']}")
    if not status["ocr"]:
        print_status("[WARN]", "OCR disabled: install Tesseract or EasyOCR.")
    if not status["qr"]:
        print_status("[WARN]", "QR disabled: install OpenCV or pyzbar.")
    if not status["yolo"]:
        print_status("[WARN]", "YOLO disabled: model failed to load.")
    if not status["vision"]:
        creds_value = resolve_credentials_path()
        if not creds_value:
            print_status(
                "[WARN]",
                "Vision disabled: GOOGLE_APPLICATION_CREDENTIALS is empty.",
            )
        else:
            print_status(
                "[WARN]",
                f"Vision disabled: credentials not found at {creds_value}.",
            )
    if not status["stego"]:
        print_status("[WARN]", "Stego disabled: install scikit-image.")


def run_scan(
    root: str,
    report_path: str,
    bip39_path: Optional[str],
    yolo_model: str,
    yolo_conf: float,
    yolo_imgsz: int,
    vision_score: float,
    use_vision: bool,
    debug: bool,
):
    if colorama_init:
        colorama_init()
    status = preflight_checks(yolo_model)
    print_preflight(status, yolo_model)
    vision_enabled = use_vision and status["vision"]
    bip39_words = load_bip39(bip39_path)
    yolo_model_instance = load_yolo(yolo_model) if status["yolo"] else None
    files = collect_images(root)
    print_status("[INFO]", f"Scanning {len(files)} image(s) in {root}")
    items: List[EvidenceItem] = []
    for path in files:
        rel_path = os.path.relpath(path, os.path.dirname(report_path))
        item = EvidenceItem(path=path, rel_path=rel_path)
        image_bgr = load_image_cv2(path)

        ocr_snippet, ocr_findings = run_ocr(image_bgr, bip39_words, status["ocr"])
        item.ocr_snippet = ocr_snippet
        item.findings.extend(ocr_findings)

        # Always try QR detection (OpenCV fallback works without pyzbar)
        qr_data, qr_findings = run_qr(image_bgr)
        item.qr_data = qr_data
        item.findings.extend(qr_findings)

        if vision_enabled:
            vision_qr, vision_objects, vision_labels, vision_findings = run_vision(
                path,
                min_score=vision_score,
                debug=debug,
            )
            # Use Vision QR if local QR detection didn't find anything
            if vision_qr and not item.qr_data:
                item.qr_data = vision_qr
            item.vision_objects = vision_objects
            item.vision_labels = vision_labels
            item.findings.extend(vision_findings)
        else:
            labels, yolo_findings = run_yolo(yolo_model_instance, path, yolo_conf, yolo_imgsz)
            item.yolo_labels = labels
            item.findings.extend(yolo_findings)

        geoint_link, geoint_findings = run_geoint(path)
        item.geoint_link = geoint_link
        item.findings.extend(geoint_findings)

        entropy, entropy_findings = run_entropy(image_bgr if status["stego"] else None, path)
        item.entropy = entropy
        item.findings.extend(entropy_findings)

        stego_payload, stego_message, stego_reason, stego_findings, stego_artifact = run_stego_decode(path)
        item.stego_payload = stego_payload
        item.stego_message = stego_message
        item.stego_reason = stego_reason
        item.findings.extend(stego_findings)
        item.stego_artifact_path = stego_artifact
        if stego_artifact:
            item.stego_artifact_thumbnail = image_to_thumbnail(stego_artifact)
            ocr_text = ocr_image_path(stego_artifact)
            item.stego_ocr = ocr_text
            # If we got text from the extracted artifact, prioritize it as the main message
            if ocr_text and not item.stego_message:
                item.stego_message = ocr_text[:300]
                item.stego_reason = f"Hidden text extracted and decoded from embedded file: {os.path.basename(stego_artifact)}"
        if stego_findings:
            for finding in stego_findings:
                print_status(
                    "[CRITICAL]",
                    f"{path} -> {finding.summary} (see report)",
                    Fore.RED if Fore else None,
                )

        if item.findings:
            print_status("[FOUND]", f"{path} -> {item.top_severity}", Fore.RED if Fore else None)
        else:
            print_status("[CLEAN]", path, Fore.GREEN if Fore else None)
        if debug:
            print_status(
                "[DEBUG]",
                f"OCR:{bool(ocr_findings)} QR:{bool(qr_data)} VISION:{len(item.vision_objects)} "
                f"LABELS:{len(item.vision_labels)} YOLO:{len(item.yolo_labels)} GEO:{bool(geoint_link)} "
                f"STEGO:{entropy} DECODE:{bool(stego_payload)}",
            )
            if item.vision_labels and not item.vision_objects:
                print_status("[DEBUG]", f"Vision labels: {', '.join(item.vision_labels[:6])}")

        items.append(item)

    generate_report(report_path, root, items)
    print_status("[DONE]", f"Report saved to {report_path}", Fore.CYAN if Fore else None)


def generate_report(report_path: str, root: str, items: List[EvidenceItem]):
    if not Environment:
        print_status("[WARN]", "Jinja2 not installed, skipping HTML report.")
        return
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    threats = sum(1 for item in items if item.findings)
    locations = sum(1 for item in items if item.geoint_link)
    for item in items:
        item.thumbnail = image_to_thumbnail(item.path)
    html = template.render(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        root=root,
        files_scanned=len(items),
        threats=threats,
        locations=locations,
        items=items,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project V.I.G.I.L. - Image Intelligence Scanner")
    parser.add_argument(
        "path",
        nargs="?",
        default="Evidence_Dump",
        help="Folder to scan (default: Evidence_Dump)",
    )
    parser.add_argument(
        "--report",
        default="report.html",
        help="Output HTML report path",
    )
    parser.add_argument(
        "--bip39",
        default=None,
        help="Path to BIP39 wordlist text file",
    )
    parser.add_argument(
        "--yolo",
        default="yolov8n.pt",
        help="YOLO model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.15,
        help="YOLO confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=960,
        help="YOLO inference size (default: 960)",
    )
    parser.add_argument(
        "--vision-score",
        type=float,
        default=0.4,
        help="Vision confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Google Vision API when available (default: true)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-image module debug info",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.path):
        print_status("[ERROR]", f"Path not found: {args.path}")
        return 1
    load_env()
    ensure_inline_credentials()
    configure_tesseract()
    run_scan(
        args.path,
        args.report,
        args.bip39,
        args.yolo,
        args.yolo_conf,
        args.yolo_imgsz,
        args.vision_score,
        args.vision,
        args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
