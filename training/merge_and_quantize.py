"""LoRA adaptörünü base modele merge edip GGUF Q4_K_M üretir.

Akış (6 adım): adaptör kontrolü -> merge (CPU, bf16) -> llama.cpp indir ->
GGUF q8_0 -> imatrix kalibrasyonu -> Q4_K_M quantize. Her adım kendi çıktısını
doğrular ve artık gerekmeyen bir önceki ara ürünü siler; yarıda kesilen bir
koşu tekrar çalıştırıldığında kaldığı yerden devam eder.

Pahalıya patlayan kararlar (yeniden keşfi maliyetli olduğu için burada):

1. Ara format f16 değil q8_0: f16 ~16GB, q8_0 ~8GB yer kaplar ve K-quant'lar
(Q4_K_M dahil) q8_0 kaynaktan kalite kaybı olmadan üretilebiliyor.
Bedeli: llama-quantize zaten kuantize bir kaynağı reddediyor, bu yüzden
`--allow-requantize` zorunlu.

2. imatrix bir EĞİTİM ADIMI DEĞİL: sadece ileri geçiş (forward pass) yapılıp
hangi ağırlık kanallarının çıktıyı daha çok etkilediği ölçülüyor, ağırlığa
dokunulmuyor. `data/train.jsonl` salt okunur — hiçbir veri dosyası
değiştirilmiyor, sadece düz metin kalibrasyon kopyası çıkarılıyor.

3. Kalibrasyon boyutu: 300 örnek denendi, llama-imatrix CPU'da 810 chunk ->
ETA ~4.5 saat çıkardı. 40 örnek ~125 chunk -> ~40 dakika; imatrix kalitesi
için bu aralık yeterli (yaygın pratikte de 100-200 chunk kullanılıyor).

4. convert_hf_to_gguf.py tek başına yetmiyor: yeni llama.cpp sürümlerinde
yanındaki `conversion/` paketinden import ediyor ve `gguf-py/` klasörünü
sys.path'e ekliyor. raw.githubusercontent'ten tek dosya indirmek bu yüzden
yeterli değil; bu iki klasör ayrıca kaynak zip'inden çekiliyor.

5. "Dosya var" != "iş bitti": yarıda kesilen araçlar diskte bozuk çıktı
bırakıyor (llama-quantize kısmi .gguf yazıp hata veriyor, llama-imatrix
birkaç chunk'ta bir kısmi checkpoint kaydediyor). Bu yüzden GGUF'lar boyut
+ magic byte ile, imatrix ise ayrı bir `.done` marker dosyasıyla
doğrulanıyor. Aksi halde bir sonraki adım "bitmiş" sanıp kendi girdisini
ssiliyor ve saatlerce iş çöpe gidiyor.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

# --- Ayarlar ---------------------------------------------------------------

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
QUANT_TYPE = "Q4_K_M"
INTERMEDIATE_TYPE = "q8_0"
CALIBRATION_SAMPLE_TARGET = 40

# --- Yollar ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

ADAPTER_DIR = OUTPUTS_DIR / "lora-llama31-8b-tr" / "final"
MERGED_DIR = OUTPUTS_DIR / "merged-bf16"
GGUF_DIR = OUTPUTS_DIR / "gguf"
LLAMA_CPP_DIR = OUTPUTS_DIR / "llama.cpp"

GGUF_INTERMEDIATE = GGUF_DIR / f"llama31-8b-tr-{INTERMEDIATE_TYPE}.gguf"
GGUF_QUANT = GGUF_DIR / f"llama31-8b-tr-{QUANT_TYPE}.gguf"

CALIBRATION_SOURCE = SCRIPT_DIR / "data" / "train.jsonl"
CALIBRATION_TXT = GGUF_DIR / "calibration.txt"
IMATRIX_FILE = GGUF_DIR / "imatrix.dat"
IMATRIX_DONE_MARKER = GGUF_DIR / "imatrix.done"

# --- llama.cpp kaynakları --------------------------------------------------

LLAMA_CPP_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
LLAMA_CPP_RAW = "https://raw.githubusercontent.com/ggml-org/llama.cpp"
LLAMA_CPP_SRC = "https://github.com/ggml-org/llama.cpp/archive/refs/tags"

CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
CONVERT_MODULES = ["conversion", "gguf-py"]  # convert script'in yanında olmalı (bkz. madde 4)
PIP_PACKAGES = ["gguf", "sentencepiece", "protobuf"]

# --- Doğrulama eşikleri (bkz. madde 5) -------------------------------------

GGUF_MAGIC = b"GGUF"
MIN_INTERMEDIATE_BYTES = 6 * 1024 ** 3  # q8_0 hedefi ~8GB, payla 6GB alt sınır
MIN_QUANT_BYTES = 3 * 1024 ** 3         # Q4_K_M hedefi ~4.9GB, payla 3GB alt sınır


# --- Yardımcılar -----------------------------------------------------------

def run(command: list, cwd: Path | None = None) -> None:
    print("$ " + " ".join(str(c) for c in command))
    result = subprocess.run([str(c) for c in command], cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}")


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}sa {minutes}dk {seconds}sn"
    if minutes:
        return f"{minutes}dk {seconds}sn"
    return f"{seconds}sn"


class Heartbeat:
    """Kendi ilerleme çıktısı olmayan uzun işlemler sırasında periyodik olarak
    'hâlâ çalışıyor' basar; 'donmuş mu?' belirsizliğini gidermek için."""

    def __init__(self, label: str, interval: float = 20.0) -> None:
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        start = time.time()
        while not self._stop.wait(self._interval):
            print(f"  ... {self._label} ({format_duration(time.time() - start)} geçti)")

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def remove_path(path: Path, reason: str) -> None:
    if not path.exists():
        return
    size = directory_size(path) if path.is_dir() else path.stat().st_size
    print(f"Cleanup: removing {path} ({human(size)}) — {reason}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_valid_gguf(path: Path, min_bytes: int) -> bool:
    """Dosyanın hem makul boyutta hem de gerçekten GGUF olduğunu doğrular."""
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(4) == GGUF_MAGIC
    except OSError:
        return False


def have_intermediate() -> bool:
    return looks_like_valid_gguf(GGUF_INTERMEDIATE, MIN_INTERMEDIATE_BYTES)


def have_final() -> bool:
    return looks_like_valid_gguf(GGUF_QUANT, MIN_QUANT_BYTES)


def have_imatrix() -> bool:
    """Marker olmadan imatrix'e güvenilmez: kısmi checkpoint olabilir."""
    return IMATRIX_FILE.exists() and IMATRIX_DONE_MARKER.exists()


def require_intermediate() -> None:
    if not have_intermediate():
        raise SystemExit(
            f"Valid intermediate GGUF not found at {GGUF_INTERMEDIATE} — rerun the "
            "script from the top so convert_to_gguf can rebuild it."
        )


def find_binary(name: str) -> Path | None:
    """llama.cpp binary'si zip'ten ya kök dizine ya da build/bin/ altına çıkar."""
    for candidate in (
        LLAMA_CPP_DIR / f"{name}.exe",
        LLAMA_CPP_DIR / name,
        LLAMA_CPP_DIR / "build" / "bin" / f"{name}.exe",
        LLAMA_CPP_DIR / "build" / "bin" / name,
    ):
        if candidate.exists():
            return candidate
    return None


def require_binary(name: str) -> Path:
    binary = find_binary(name)
    if binary is None:
        raise SystemExit(f"{name} not found under {LLAMA_CPP_DIR}")
    return binary


# --- Adım 1: adaptör kontrolü ----------------------------------------------

def check_adapter() -> None:
    if not ADAPTER_DIR.is_dir():
        raise SystemExit(f"Adapter not found at {ADAPTER_DIR}\nRun train.py first.")

    print(f"Adapter found : {ADAPTER_DIR}")
    print(f"Size          : {human(directory_size(ADAPTER_DIR))}")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)


# --- Adım 2: merge ---------------------------------------------------------

def merge_adapter() -> None:
    if have_intermediate() or have_final():
        print("Downstream GGUF already exists and looks valid, merge no longer needed, skipping")
        return

    if MERGED_DIR.is_dir() and any(MERGED_DIR.glob("*.safetensors")):
        print(f"Merged model already exists, skipping: {MERGED_DIR}")
        print(f"Size: {human(directory_size(MERGED_DIR))}")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model on CPU: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Applying adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    with Heartbeat("model birleştiriliyor / diske yazılıyor"):
        print("Merging weights")
        model = model.merge_and_unload()

        print(f"Saving merged model: {MERGED_DIR}")
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

        AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(str(MERGED_DIR))

    del model
    print(f"Merge complete. Size: {human(directory_size(MERGED_DIR))}")


# --- Adım 3: llama.cpp ------------------------------------------------------

def latest_release() -> tuple[str, str]:
    """En son Windows CPU build'inin (tag, indirme url'i) çiftini döner."""
    request = urllib.request.Request(
        LLAMA_CPP_API + "?per_page=10",
        headers={"User-Agent": "merge-and-quantize"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        for release in json.load(response):
            for asset in release["assets"]:
                name = asset["name"].lower()
                if "bin-win-cpu-x64" in name and name.endswith(".zip"):
                    return release["tag_name"], asset["browser_download_url"]
    raise SystemExit("No Windows CPU build found in recent llama.cpp releases")


def download(url: str, destination: Path) -> None:
    print(f"$ download {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "merge-and-quantize"})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"  saved {human(len(payload))} -> {destination}")


def fetch_conversion_modules(tag: str) -> None:
    """conversion/ ve gguf-py/ klasörlerini aynı tag'in kaynak zip'inden çeker
    (bkz. madde 4). Tüm repo inip sadece bu iki klasör (~2MB) çıkarılır."""
    archive = LLAMA_CPP_DIR / f"source-{tag}.zip"
    download(f"{LLAMA_CPP_SRC}/{tag}.zip", archive)

    print(f"$ extracting {', '.join(CONVERT_MODULES)} from source archive")
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        root_prefix = members[0].split("/")[0] + "/"
        for member in members:
            if member.endswith("/") or not member.startswith(root_prefix):
                continue
            relative = member[len(root_prefix):]
            if not any(relative.startswith(d + "/") for d in CONVERT_MODULES):
                continue
            target = LLAMA_CPP_DIR / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
            extracted += 1

    archive.unlink()
    print(f"  extracted {extracted} files")

    missing = [d for d in CONVERT_MODULES if not (LLAMA_CPP_DIR / d).is_dir()]
    if missing:
        raise SystemExit(f"Could not extract required module(s) from source archive: {missing}")


def fetch_llama_cpp() -> None:
    have_binaries = all(find_binary(n) for n in ("llama-quantize", "llama-imatrix"))
    have_modules = all((LLAMA_CPP_DIR / d).is_dir() for d in CONVERT_MODULES)

    if have_binaries and CONVERT_SCRIPT.exists() and have_modules:
        print("llama.cpp already present, skipping download")
        return

    LLAMA_CPP_DIR.mkdir(parents=True, exist_ok=True)

    tag, binary_url = latest_release()
    print(f"Release: {tag}")

    if not have_binaries:
        archive = LLAMA_CPP_DIR / "llama-bin-win-cpu-x64.zip"
        download(binary_url, archive)
        print(f"$ extract {archive.name}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(LLAMA_CPP_DIR)
        archive.unlink()

    if not CONVERT_SCRIPT.exists():
        download(f"{LLAMA_CPP_RAW}/{tag}/{CONVERT_SCRIPT.name}", CONVERT_SCRIPT)

    if not have_modules:
        fetch_conversion_modules(tag)

    run([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES])

    require_binary("llama-quantize")
    require_binary("llama-imatrix")


# --- Adım 4: GGUF'a çevirme -------------------------------------------------

def convert_to_gguf() -> None:
    if have_final():
        print("Quantized GGUF already exists and looks valid, conversion no longer needed, skipping")
        return

    if have_intermediate():
        print(f"{INTERMEDIATE_TYPE} GGUF already exists and looks valid, skipping: {GGUF_INTERMEDIATE}")
        print(f"Size: {human(GGUF_INTERMEDIATE.stat().st_size)}")
        return

    remove_path(GGUF_INTERMEDIATE, "corrupt/incomplete leftover from a previous failed run")

    if not MERGED_DIR.is_dir() or not any(MERGED_DIR.glob("*.safetensors")):
        raise SystemExit(
            f"Merged model not found at {MERGED_DIR} — rerun the script from the top "
            "so merge_adapter can rebuild it."
        )
    if not CONVERT_SCRIPT.exists():
        raise SystemExit(f"Conversion script not found: {CONVERT_SCRIPT}")

    run([
        sys.executable,
        CONVERT_SCRIPT,
        MERGED_DIR,
        "--outfile", GGUF_INTERMEDIATE,
        "--outtype", INTERMEDIATE_TYPE,
    ])

    if not have_intermediate():
        raise SystemExit(f"Conversion produced an invalid/truncated file: {GGUF_INTERMEDIATE}")

    print(f"Size: {human(GGUF_INTERMEDIATE.stat().st_size)}")
    remove_path(MERGED_DIR, "already baked into the GGUF, no longer needed")


# --- Adım 5: imatrix --------------------------------------------------------

def build_calibration_file() -> None:
    """train.jsonl'dan düzenli aralıklarla örnek seçip düz metin kalibrasyon
    dosyası yazar. Kaynak dosya salt okunur (bkz. madde 2)."""
    if CALIBRATION_TXT.exists():
        print(f"Calibration file already exists, skipping: {CALIBRATION_TXT}")
        return

    if not CALIBRATION_SOURCE.exists():
        raise SystemExit(f"Calibration source not found: {CALIBRATION_SOURCE}")

    lines = CALIBRATION_SOURCE.read_text(encoding="utf-8").splitlines()
    stride = max(1, len(lines) // CALIBRATION_SAMPLE_TARGET)
    sampled = lines[::stride]
    print(f"Sampling {len(sampled)}/{len(lines)} examples (stride={stride}) from {CALIBRATION_SOURCE}")

    # Önce .tmp'ye yazılıp sonra rename ediliyor: yarım kalan bir yazım
    # asla "hazır dosya" gibi görünmesin.
    tmp_path = CALIBRATION_TXT.with_name(CALIBRATION_TXT.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for raw_line in sampled:
            raw_line = raw_line.strip()
            if raw_line:
                out.write(json.loads(raw_line)["text"])
                out.write("\n\n")
    tmp_path.replace(CALIBRATION_TXT)

    print(f"Calibration file written: {CALIBRATION_TXT} ({human(CALIBRATION_TXT.stat().st_size)})")


def generate_imatrix() -> None:
    if have_final():
        print("Quantized GGUF already exists and looks valid, imatrix no longer needed, skipping")
        return

    if have_imatrix():
        print(f"imatrix already exists and is marked complete, skipping: {IMATRIX_FILE}")
        return

    # Marker yoksa elde kalan şey yarıda kesilmiş bir checkpoint'tir: güvenilmez.
    remove_path(IMATRIX_FILE, "incomplete (no completion marker) — leftover from an interrupted run")
    remove_path(IMATRIX_DONE_MARKER, "stale marker without a matching imatrix file")

    require_intermediate()
    binary = require_binary("llama-imatrix")
    build_calibration_file()

    run([
        binary,
        "-m", GGUF_INTERMEDIATE,
        "-f", CALIBRATION_TXT,
        "-o", IMATRIX_FILE,
        "--output-format", "dat",
    ])

    if not IMATRIX_FILE.exists() or IMATRIX_FILE.stat().st_size == 0:
        raise SystemExit(f"imatrix generation produced an empty/missing file: {IMATRIX_FILE}")

    IMATRIX_DONE_MARKER.write_text("ok", encoding="utf-8")
    print(f"imatrix written: {IMATRIX_FILE} ({human(IMATRIX_FILE.stat().st_size)})")


# --- Adım 6: quantize -------------------------------------------------------

def quantize() -> None:
    if have_final():
        print(f"Quantized GGUF already exists and looks valid, skipping: {GGUF_QUANT}")
    else:
        remove_path(GGUF_QUANT, "corrupt/incomplete leftover from a previous failed run")

        require_intermediate()
        if not have_imatrix():
            raise SystemExit(
                f"No complete imatrix found ({IMATRIX_FILE}). Run generate_imatrix first "
                "(rerun the script from the top)."
            )
        binary = require_binary("llama-quantize")

        # Kaynak q8_0, yani zaten kuantize (bkz. madde 1) — llama-quantize
        # böyle bir kaynağı açıkça izin verilmedikçe reddediyor.
        run([
            binary,
            "--allow-requantize",
            "--imatrix", IMATRIX_FILE,
            GGUF_INTERMEDIATE, GGUF_QUANT, QUANT_TYPE,
        ])

        if not have_final():
            raise SystemExit(
                f"Quantize command exited successfully but the output failed validation: {GGUF_QUANT}. "
                "Do not trust or ship this file."
            )

    remove_path(GGUF_INTERMEDIATE, "already quantized into the final GGUF, no longer needed")
    remove_path(IMATRIX_FILE, "already baked into the final GGUF, no longer needed")
    remove_path(IMATRIX_DONE_MARKER, "no longer needed once the final GGUF exists")
    remove_path(CALIBRATION_TXT, "no longer needed after imatrix generation")


# --- Sonuç ------------------------------------------------------------------

def report() -> None:
    if not have_final():
        raise SystemExit(f"Final file failed validation, refusing to report success: {GGUF_QUANT}")

    print()
    print("=" * 66)
    print(" DONE")
    print("=" * 66)
    print(f"File   : {GGUF_QUANT}")
    print(f"Size   : {human(GGUF_QUANT.stat().st_size)}")
    print("Hashing (this takes a moment)")
    print(f"SHA256 : {sha256_of(GGUF_QUANT)}")
    print()
    print("Download this file to the inference machine.")
    print()
    print("Intermediate files (merged model, intermediate GGUF, imatrix) were")
    print("already cleaned up automatically during the run.")
    if LLAMA_CPP_DIR.exists():
        print(f"You may also delete {LLAMA_CPP_DIR} ({human(directory_size(LLAMA_CPP_DIR))}) to free space.")


def main() -> None:
    pipeline = [
        ("Checking adapter", check_adapter),
        ("Merging LoRA adapter into base model", merge_adapter),
        ("Fetching llama.cpp (prebuilt, no compiler needed)", fetch_llama_cpp),
        (f"Converting merged model to GGUF ({INTERMEDIATE_TYPE})", convert_to_gguf),
        ("Generating imatrix (quantization calibration)", generate_imatrix),
        (f"Quantizing to {QUANT_TYPE}", quantize),
    ]

    overall_start = time.time()
    for number, (title, action) in enumerate(pipeline, start=1):
        print()
        print("=" * 66)
        print(f" STEP {number}/{len(pipeline)}  {title}")
        print("=" * 66)

        step_start = time.time()
        action()
        print(f"(bu adım {format_duration(time.time() - step_start)} sürdü, "
            f"toplam geçen süre: {format_duration(time.time() - overall_start)})")

    report()


if __name__ == "__main__":
    main()
