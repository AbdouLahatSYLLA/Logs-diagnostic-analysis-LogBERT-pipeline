"""Generate logbert_v3_bgl.ipynb : reprise V2 + BGL + multi-seed."""
import json
import re

v2 = json.load(open("logbert_hdfs_v2_resultats.ipynb"))
v2_export = "".join(v2["cells"][16]["source"])
m = re.search(r'inference_source = r""".*?"""\n', v2_export, re.S)
assert m, "inference_source block not found in V2 export cell"
INFERENCE_BLOCK = m.group(0)
assert "def load_logbert_bundle" in INFERENCE_BLOCK and "def predict_event_sequences" in INFERENCE_BLOCK

cells = []

def md(cid, text):
    cells.append({"id": cid, "cell_type": "markdown", "metadata": {},
                  "source": [l + "\n" for l in text.split("\n")]})

def code(cid, text):
    cells.append({"id": cid, "cell_type": "code", "metadata": {},
                  "source": [l + "\n" for l in text.split("\n")],
                  "outputs": [], "execution_count": None})

md("v3-title", """# LogBERT V3 — généralisation BGL + rigueur multi-seed

V3 reprend V2 à l'identique (même dépôt officiel au commit `a845e61`, même scoring paper-style top-`g`/`r`, même protocole : train normal-only, sélection sur validation, test verrouillé) et ajoute deux exigences : (1) un second dataset de nature différente, **BGL** (supercalculateur BlueGene/L, fenêtres temporelles de 5 min, séquences longues ~562 vs ~19 pour HDFS) ; (2) la répétition **multi-seed** à splits figés, avec moyenne ± écart-type.

Réglages officiels BGL : `m=0.5`, fenêtre 5 min / pas 1 min, `train_ratio=0.4` (shuffle seed 12), 200 époques max + arrêt précoce 10, batch 32, `seq_len=512`, `min_len=10`, 15 candidats. V3 teste `m ∈ {0.35, 0.50, 0.65}` autour du réglage officiel, puis répète le `m` sélectionné sur 3 seeds.

Baseline article BGL (Guo et al., 2021) : Precision 89.40 %, Recall 92.32 %, F1 90.83 % (référence indicative non appariée).

Exécution par phases, avec reprise des checkpoints existants : **Phase A** (ablation `m`, seed 1234), **Phase B** (multi-seed sur le `m` sélectionné), **Phase C** (optionnelle : seeds HDFS supplémentaires, désactivée par défaut).""")

md("v3-deps-title", """## 1. Dépendances et dépôt officiel

Même base que V2 : on garde le Torch de Kaggle, on fixe pandas compatible NumPy 2.x, on clone le dépôt officiel au commit gelé. GPU requis.""")

code("v3-deps", """# Garder la version Torch fournie par Kaggle; installer seulement les dépendances de parsing/évaluation.
%pip install -q --force-reinstall --no-deps pandas==2.2.3
%pip install -q --no-deps regex==2023.12.25
print('Dépendances installées. Si NumPy/Pandas étaient déjà importés, redémarrer le kernel Kaggle avant de poursuivre.')

import os, sys, json, ast, random, shutil, subprocess, importlib.util
from pathlib import Path
import numpy as np
if int(np.__version__.split(".")[0]) < 2:
    raise RuntimeError(f"NumPy {np.__version__} requis en 2.x avec ce notebook.")
print("Runtime NumPy", np.__version__)
import pandas as pd
import torch
REQUIRE_GPU = True
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("GPU non détecté. Dans Kaggle : Settings > Accelerator > GPU, puis relancer le notebook.")
print("GPU actif :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun (bloqué)")
for _name, _value in {"float": float, "int": int, "Inf": np.inf}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)
if "torch._six" not in sys.modules:
    import types
    _torch_six = types.ModuleType("torch._six")
    _torch_six.string_classes = (str,)
    _torch_six.int_classes = (int,)
    sys.modules["torch._six"] = _torch_six
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

BASE_SEED = 1234
random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(BASE_SEED)

REPO_URL = "https://github.com/HelenGuohx/LogBERT.git"
REPO_COMMIT = "a845e612e919a6930065b69a40098eaaac26e1a3"
WORK_DIR = Path(os.environ.get("LOGBERT_WORK_DIR", str(Path.cwd() / ".logbert_v3_bgl"))).expanduser().resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
REPO_DIR = WORK_DIR / "LogBERT"
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", REPO_COMMIT], check=True)
sys.path.insert(0, str(REPO_DIR))
print("Repository:", REPO_DIR, "| commit:", subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip())
print("Torch:", torch.__version__)

# V3 protocol constants (frozen before any training).
V3_M_GRID = [0.35, 0.50, 0.65]          # around the official BGL default m=0.50
V3_SEEDS = [1234, 2026, 999]            # Phase A uses 1234; Phase B adds the other two
V3_G_GRID_FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
V3_R_GRID = np.arange(0.0, 0.801, 0.01)
V3_TARGET_PREVALENCE = 2630 / (10045 + 2630)   # paper BGL test prevalence (~20.75 %)
PAPER_BGL = {"version": "Article LogBERT BGL", "Precision": .8940, "Recall": .9232, "F1": .9083,
             "ROC_AUC": float("nan"), "PR_AUC": float("nan")}
HDFS_V2_REF = {"version": "V2 HDFS m=.75,g=8,r=.04", "Precision": .8267, "Recall": .8362,
               "F1": .8314, "ROC_AUC": .9235, "PR_AUC": .7051}
RUN_PHASE_A = True    # m ablation, seed 1234
RUN_PHASE_B = True    # multi-seed on selected m (needs Phase A selection)
RUN_PHASE_C = True   # extra HDFS seeds; needs HDFS files from the V2 session
print("V3 config:", {"m_grid": V3_M_GRID, "seeds": V3_SEEDS, "target_prevalence": round(V3_TARGET_PREVALENCE, 4)})""")

md("v3-data-title", """## 2. Trouver ou télécharger BGL

Le notebook cherche d'abord un `BGL.log` attaché (`BGL_LOG_PATH`, `/kaggle/input`, `/content`). Sinon il télécharge l'archive officielle LogHub `BGL.zip` (57 489 019 octets, MD5 vérifié) et en extrait `BGL.log`. BGL porte ses labels dans le log lui-même (préfixe `-` = normal), aucun fichier de labels séparé n'est requis.""")

code("v3-data", """explicit_bgl = os.environ.get("BGL_LOG_PATH")
if explicit_bgl:
    BGL_LOG = Path(explicit_bgl).expanduser().resolve()
else:
    roots = [Path("/kaggle/input"), Path("/content"), Path.cwd()]
    if os.environ.get("BGL_INPUT_DIR"):
        roots.insert(0, Path(os.environ["BGL_INPUT_DIR"]).expanduser())
    roots = list(dict.fromkeys(r.resolve() for r in roots if r.exists()))
    found = [p for root in roots for p in root.rglob("*")
             if p.is_file() and p.name == "BGL.log" and p.stat().st_size >= 100_000_000]
    BGL_LOG = found[0] if found else None
    if BGL_LOG is None:
        import hashlib, urllib.request, zipfile
        download_dir = WORK_DIR / "downloaded_bgl"
        download_dir.mkdir(parents=True, exist_ok=True)
        archive_path = download_dir / "BGL.zip"
        bgl_target = download_dir / "BGL.log"
        archive_url = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
        expected_size, expected_md5 = 57489019, "4452953c470f2d95fcb32d5f6e733f7a"
        if not bgl_target.exists():
            print("BGL.log introuvable localement ; téléchargement LogHub officiel (~57 Mo)...")
            digest = hashlib.md5()
            request = urllib.request.urlopen(archive_url, timeout=120)
            with archive_path.open("wb") as output:
                while True:
                    chunk = request.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
            if archive_path.stat().st_size != expected_size or digest.hexdigest() != expected_md5:
                raise IOError("Archive BGL.zip non conforme (taille ou MD5).")
            with zipfile.ZipFile(archive_path) as archive:
                member = next(n for n in archive.namelist() if n.endswith("BGL.log"))
                with archive.open(member) as src, bgl_target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            print("BGL.log extrait :", bgl_target, f"{bgl_target.stat().st_size / 1e9:.2f} Go")
        BGL_LOG = bgl_target
print("BGL.log :", BGL_LOG, f"{BGL_LOG.stat().st_size / 1e9:.2f} Go")""")

md("v3-parse-title", """## 3. Parsing Drain officiel BGL et sessionnisation (fenêtres 5 min)

On appelle `parse_log` du dépôt officiel (Drain `depth=3`, `st=0.3`, masques hexadécimaux/IP/numériques), puis on réplique exactement le bloc `__main__` officiel : labels (`-` = normal), fenêtres glissantes 5 min / pas 1 min via `logdeep`, shuffle seed 12, `train_ratio=0.4`. Le split officiel train/test est conservé ; on y ajoute une validation explicite (10 % des normales restantes, 20 % des anomalies), disjointe du train et du test, pour la sélection sans fuite.""")

code("v3-parse", """REPO_BGL = REPO_DIR / "BGL"
OUTPUT_BGL = REPO_DIR / "output" / "bgl"
OUTPUT_BGL.mkdir(parents=True, exist_ok=True)
(OUTPUT_BGL / "bert").mkdir(parents=True, exist_ok=True)
print("Contenu BGL officiel :", sorted(p.name for p in REPO_BGL.iterdir()))

DATA_HOME = Path.home() / ".dataset" / "bgl"
DATA_HOME.mkdir(parents=True, exist_ok=True)
target = DATA_HOME / "BGL.log"
if target.exists() or target.is_symlink():
    target.unlink()
target.symlink_to(BGL_LOG)

spec_bgl = importlib.util.spec_from_file_location("bgl_data_process", REPO_BGL / "data_process.py")
dp_bgl = importlib.util.module_from_spec(spec_bgl)
spec_bgl.loader.exec_module(dp_bgl)
from logdeep.dataset.session import sliding_window

# Official Drain parsing (4.7 M lines: this is the long step, run once).
dp_bgl.parse_log(str(DATA_HOME) + "/", str(OUTPUT_BGL) + "/", "BGL.log", "drain")

STRUCTURED_BGL = OUTPUT_BGL / "BGL.log_structured.csv"
assert STRUCTURED_BGL.exists(), "Parsing BGL sans sortie structurée."
df = pd.read_csv(STRUCTURED_BGL)
print("Structured:", len(df), "lignes,", df.EventId.nunique(), "clés brutes.")

# Exact official __main__ transformation, then our explicit validation split.
df["datetime"] = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f")
df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
df["timestamp"] = df["datetime"].values.astype(np.int64) // 10 ** 9
df["deltaT"] = (df["datetime"].diff() / np.timedelta64(1, "s")).fillna(0)
windows = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]],
                         para={"window_size": 5 * 60, "step_size": 1 * 60})
print("Fenêtres 5 min / pas 1 min :", len(windows))

normal_all = windows[windows.Label == 0].sample(frac=1, random_state=12).reset_index(drop=True)
anomaly_all = windows[windows.Label == 1].reset_index(drop=True)
BGL_TRAIN_RATIO = 0.4  # official BGL setting
train_len = int(len(normal_all) * BGL_TRAIN_RATIO)
bgl_train = normal_all.iloc[:train_len].reset_index(drop=True)
remaining_normal = normal_all.iloc[train_len:].reset_index(drop=True)
n_val_n = int(len(remaining_normal) * 0.10)
n_val_a = int(len(anomaly_all) * 0.20)
bgl_val_normal = remaining_normal.iloc[:n_val_n].reset_index(drop=True)
bgl_test_normal = remaining_normal.iloc[n_val_n:].reset_index(drop=True)
bgl_val_anomaly = anomaly_all.iloc[:n_val_a].reset_index(drop=True)
bgl_test_anomaly = anomaly_all.iloc[n_val_a:].reset_index(drop=True)

MIN_LEN = 10  # official predictor setting, same as V2
def keep_eligible(frame):
    return frame[frame.EventId.map(len) >= MIN_LEN].reset_index(drop=True)
bgl_train, bgl_val_normal, bgl_test_normal = map(keep_eligible, [bgl_train, bgl_val_normal, bgl_test_normal])
bgl_val_anomaly, bgl_test_anomaly = map(keep_eligible, [bgl_val_anomaly, bgl_test_anomaly])
assert (bgl_train.Label == 0).all()

def write_bgl_sequences(frame, filename):
    path = OUTPUT_BGL / filename
    with path.open("w", encoding="utf-8") as stream:
        for events in frame.EventId:
            stream.write(" ".join(map(str, events)) + " \\n")
    return path

for frame, name in [(bgl_train, "train"), (bgl_val_normal, "val_normal"), (bgl_val_anomaly, "val_abnormal"),
                    (bgl_test_normal, "test_normal"), (bgl_test_anomaly, "test_abnormal")]:
    write_bgl_sequences(frame, name)

def v3_file_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)
V3_BGL_FILES = {name: OUTPUT_BGL / name for name in ["train", "val_normal", "val_abnormal", "test_normal", "test_abnormal"]}
V3_BGL_COUNTS = {name: v3_file_lines(path) for name, path in V3_BGL_FILES.items()}
print("Train normal=%d ; val %d/%d ; test %d/%d" % (V3_BGL_COUNTS["train"], V3_BGL_COUNTS["val_normal"],
      V3_BGL_COUNTS["val_abnormal"], V3_BGL_COUNTS["test_normal"], V3_BGL_COUNTS["test_abnormal"]))
print("Prévalence test BGL : %.2f %%" % (100 * V3_BGL_COUNTS["test_abnormal"] / (V3_BGL_COUNTS["test_normal"] + V3_BGL_COUNTS["test_abnormal"])))
lengths = windows.EventId.map(len)
print("Longueur médiane / p95 des fenêtres :", int(lengths.median()), int(lengths.quantile(.95)))
(WORK_DIR / "v3_bgl_split.json").write_text(json.dumps({"counts": V3_BGL_COUNTS, "repo_commit": REPO_COMMIT}, indent=2))""")

md("v3-train-title", """## Phase A — Ablation contrôlée de `m` sur BGL (seed 1234)

Même grille d'esprit que V2, centrée sur le défaut officiel BGL (`m=0.5`). Entraînement repris si le checkpoint existe déjà. Le vocabulaire vient du seul train.""")

code("v3-train", """import random
import subprocess
import matplotlib.pyplot as plt

# Bulletproof NumPy-2 ragged fix: explicit object-array construction.
# np.array(ragged, dtype=object) still raises inhomogeneous-shape on NumPy 2.x
# when elements are ndarrays (BGL long windows) — plain element assignment never does.
import re as _re
_OBJECT_TMP = {"logkey_seq_pairs": "_tmp_logkey", "time_seq_pairs": "_tmp_time",
               "log_seqs": "_tmp_logseq", "tim_seqs": "_tmp_timseq"}
_PATCH_FILES = {
    REPO_DIR / "bert_pytorch" / "dataset" / "sample.py": ["logkey_seq_pairs", "time_seq_pairs"],
    REPO_DIR / "bert_pytorch" / "predict_log.py": ["log_seqs", "tim_seqs"],
}
for _path, _vars in _PATCH_FILES.items():
    _text = _path.read_text(encoding="utf-8")
    for _var in _vars:
        _tmp = _OBJECT_TMP[_var]
        if f"np.empty(len({_tmp}), dtype=object)" in _text:
            continue  # already patched (idempotent across re-runs)
        _pattern = _re.compile(rf"^([ \t]*){_var} = np\.array\({_var}(?:, dtype=object)?\)[ \t]*$", _re.M)
        _match = _pattern.search(_text)
        if not _match:
            raise RuntimeError(f"Patch target not found: {_var} in {_path.name}")
        _ind = _match.group(1)
        _block = chr(10).join([
            f"{_ind}{_tmp} = list({_var})",
            f"{_ind}{_var} = np.empty(len({_tmp}), dtype=object)",
            f"{_ind}for _i, _s in enumerate({_tmp}):",
            f"{_ind}    {_var}[_i] = _s",
            f"{_ind}del {_tmp}",
        ])
        _text = _pattern.sub(_block, _text, count=1)
    _path.write_text(_text, encoding="utf-8")
print("Ragged-array patches applied (explicit object construction, sample.py + predict_log.py).")
# Without this, an already-imported bert_pytorch (previous failed run) would shadow
# the patched files and replay the old bytecode. Purge, then import fresh.
for _mod in [m for m in list(sys.modules) if m == "bert_pytorch" or m.startswith("bert_pytorch.")]:
    del sys.modules[_mod]
print("Stale bert_pytorch modules purged:", [m for m in sys.modules if m.startswith("bert_pytorch")])
from bert_pytorch import Trainer
from bert_pytorch.dataset.utils import seed_everything

from bert_pytorch.dataset import WordVocab
v3_vocab_path = OUTPUT_BGL / "vocab.pkl"
if (OUTPUT_BGL / "train").exists():
    subprocess.run([sys.executable, "logbert.py", "vocab"], cwd=REPO_BGL, check=True)
    v3_vocab = WordVocab.load_vocab(str(v3_vocab_path))
    print("Vocabulaire BGL (train only) :", len(v3_vocab))
    spec_v3 = importlib.util.spec_from_file_location("bgl_logbert_v3_config", REPO_BGL / "logbert.py")
    cfg_v3 = importlib.util.module_from_spec(spec_v3)
    spec_v3.loader.exec_module(cfg_v3)
    V3_G_GRID = [g for g in V3_G_GRID_FULL if g <= len(v3_vocab)]
    print("Grille g retenue :", V3_G_GRID)
else:
    v3_vocab, cfg_v3, V3_G_GRID = None, None, list(V3_G_GRID_FULL)
    print("Fichiers BGL absents : vocab/config BGL sautes (reprise Phase C seule).")

def v3_train_model(seed, mask_ratio, base_options=None):
    \"\"\"Train (or reload) one BGL model. Splits stay fixed; only init/training vary with seed.\"\"\"
    tag = f"m{mask_ratio:.2f}".replace(".", "p")
    model_dir = OUTPUT_BGL / "v3" / f"seed{seed}" / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    options = (base_options or cfg_v3.options).copy()
    options.update({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": str(OUTPUT_BGL) + "/",
        "model_dir": str(model_dir) + "/",
        "model_path": str(model_dir / "best_bert.pth"),
        "vocab_path": str(v3_vocab_path),
        "scale_path": str(model_dir / "scale.pkl"),
        "mask_ratio": mask_ratio,
        "num_workers": 2,
        "on_memory": True,
    })
    seed_everything(seed=seed)
    checkpoint = Path(options["model_path"])
    if checkpoint.exists():
        print(f"Checkpoint existant seed={seed} m={mask_ratio:.2f} : reprise sans réentraînement.")
    else:
        print(f"\\n=== V3 training seed={seed} m={mask_ratio:.2f} (séquences longues : étape coûteuse) ===")
        trainer = Trainer(options)
        trainer.train()
        del trainer
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint manquant seed={seed} m={mask_ratio}")
    try:
        model = torch.load(str(checkpoint), map_location=options["device"], weights_only=False)
    except TypeError:
        model = torch.load(str(checkpoint), map_location=options["device"])
    model.to(options["device"]); model.eval()
    return model, options

V3A_MODELS, V3A_OPTIONS = {}, {}
if RUN_PHASE_A:
    for mask_ratio in V3_M_GRID:
        model, options = v3_train_model(V3_SEEDS[0], mask_ratio)
        V3A_MODELS[mask_ratio] = model
        V3A_OPTIONS[mask_ratio] = options
        del model
    print("Phase A terminée :", sorted(V3A_OPTIONS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = ["#315A8A", "#D07845", "#4E9A75"]
    for index, mask_ratio in enumerate(V3_M_GRID):
        log_dir = OUTPUT_BGL / "v3" / f"seed{V3_SEEDS[0]}" / f"m{mask_ratio:.2f}".replace(".", "p")
        train_log = pd.read_csv(log_dir / "train_log2.csv")
        valid_log = pd.read_csv(log_dir / "valid_log2.csv")
        axes[0].plot(train_log.epoch, train_log.loss, color=colors[index % 3], label=f"m={mask_ratio:.2f}")
        axes[1].plot(valid_log.epoch, valid_log.loss, color=colors[index % 3], label=f"m={mask_ratio:.2f}")
    axes[0].set_title("Objectif train"); axes[1].set_title("Objectif validation interne")
    for ax in axes:
        ax.set_xlabel("Époque"); ax.grid(alpha=.22); ax.legend(frameon=False)
    axes[0].set_ylabel("MLKP + contribution VHM")
    fig.suptitle("V3 BGL — courbes d'apprentissage selon m (seed 1234)")
    fig.tight_layout(); fig.savefig(WORK_DIR / "logbert_v3_loss_curves.png", dpi=220, bbox_inches="tight"); plt.show()
else:
    print("RUN_PHASE_A=False : ablation sautée.")""")

md("v3-select-title", """## Phase A — Sélection validation-only de `m`, `g`, `r`, puis test verrouillé BGL

Même scoring paper-style que V2 (erreurs hors top-`g` indépendantes, `r` en proportion). Seul le triplet sélectionné sur validation est scoré sur le test BGL.""")

code("v3-select", """from bert_pytorch.dataset import LogDataset
from bert_pytorch import Predictor
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from scipy.stats import norm

def v3_scores_by_g(model, options, vocab, filename, seed, g_grid):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    predictor = Predictor(options)
    log_seqs, time_seqs = predictor.generate_test(
        options["output_dir"], filename, predictor.window_size,
        predictor.adaptive_window, predictor.seq_len, None, predictor.min_len)
    dataset = LogDataset(log_seqs, time_seqs, vocab, seq_len=predictor.seq_len,
                         corpus_lines=predictor.corpus_lines, on_memory=False,
                         predict_mode=True, mask_ratio=options["mask_ratio"])
    loader = DataLoader(dataset, batch_size=predictor.batch_size,
                        num_workers=predictor.num_workers, collate_fn=dataset.collate_fn)
    errors = {g: [] for g in g_grid}
    n_masked_by_sequence = []
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(options["device"]) for key, value in batch.items()}
        with torch.no_grad():
            logits = model(batch["bert_input"], batch["time_input"])["logkey_output"]
        labels = batch["bert_label"]
        for i in range(len(labels)):
            mask = labels[i] > 0
            target = labels[i][mask]
            masked_logits = logits[i][mask]
            n_masked = len(target)
            n_masked_by_sequence.append(n_masked)
            if n_masked:
                top_ids = torch.topk(masked_logits, k=max(g_grid), dim=-1).indices
                matches = top_ids == target.unsqueeze(1)
                for g in g_grid:
                    errors[g].append(int((~matches[:, :g].any(dim=1)).sum().item()))
            else:
                for g in g_grid:
                    errors[g].append(0)
        if batch_index and batch_index % 500 == 0 and batch_index:
            print(f"{filename}: {sum(n_masked_by_sequence):,} masked positions")
    denominator = np.maximum(np.asarray(n_masked_by_sequence, dtype=float), 1)
    return {g: np.asarray(errors[g], dtype=float) / denominator for g in g_grid}

def v3_threshold_metrics(y, score, r, prior):
    pred = (score > r).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision_target = prior * recall / max(prior * recall + (1 - prior) * fpr, 1e-12)
    f1_target = 2 * precision_target * recall / max(precision_target + recall, 1e-12)
    return {"Precision_target": precision_target, "Recall": recall, "F1_target": f1_target,
            "FPR": fpr, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}

def v3_wilson(successes, total, confidence=0.95):
    if total <= 0:
        return [float("nan"), float("nan")]
    z = norm.ppf(.5 + confidence / 2); p = successes / total; den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0., center - half), min(1., center + half)]

def v3_select_triplet(model_by_m, options_by_m, vocab, counts, g_grid, r_grid, prior):
    y_val = np.r_[np.zeros(counts["val_normal"], dtype=int), np.ones(counts["val_abnormal"], dtype=int)]
    rows, scores = [], {}
    for m, model in model_by_m.items():
        opts = options_by_m[m]
        val_n = v3_scores_by_g(model, opts, vocab, "val_normal", 1234, g_grid)
        val_a = v3_scores_by_g(model, opts, vocab, "val_abnormal", 1235, g_grid)
        scores[m] = (val_n, val_a)
        for g in g_grid:
            score = np.r_[val_n[g], val_a[g]]
            for r in r_grid:
                rows.append({"m": m, "g": g, "r": float(r), **v3_threshold_metrics(y_val, score, r, prior)})
    search = pd.DataFrame(rows)
    best = search.sort_values(["F1_target", "Recall", "Precision_target"], ascending=False).iloc[0]
    return {"m": float(best.m), "g": int(best.g), "r": float(best.r),
            "validation_F1_target": float(best.F1_target),
            "validation_precision_target": float(best.Precision_target),
            "validation_recall": float(best.Recall)}, search

def v3_score_locked_test(model, options, vocab, g, r, counts):
    y_test = np.r_[np.zeros(counts["test_normal"], dtype=int), np.ones(counts["test_abnormal"], dtype=int)]
    test_n = v3_scores_by_g(model, options, vocab, "test_normal", 1236, [g])
    test_a = v3_scores_by_g(model, options, vocab, "test_abnormal", 1237, [g])
    scores = np.r_[test_n[g], test_a[g]]
    pred = (scores > r).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {"m": float(options["mask_ratio"]), "g": int(g), "r": float(r),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "FPR": float(fp / max(tn + fp, 1)), "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
            "ROC_AUC": float(roc_auc_score(y_test, scores)),
            "PR_AUC": float(average_precision_score(y_test, scores)),
            "test_prevalence": float(y_test.mean()),
            "Precision_CI95": v3_wilson(tp, tp + fp), "Recall_CI95": v3_wilson(tp, tp + fn)}, scores, y_test

if RUN_PHASE_A:
    V3A_MODELS, V3A_OPTIONS = {}, {}
    for m in V3_M_GRID:
        _model, _options = v3_train_model(V3_SEEDS[0], m)
        V3A_MODELS[m], V3A_OPTIONS[m] = _model, _options
        del _model
    V3A_SELECTION, v3a_search = v3_select_triplet(V3A_MODELS, V3A_OPTIONS, v3_vocab, V3_BGL_COUNTS,
                                                  V3_G_GRID, V3_R_GRID, V3_TARGET_PREVALENCE)
    print("V3 Phase A sélection validation :", V3A_SELECTION)
    v3a_search.to_csv(WORK_DIR / "v3_bgl_search_seed1234.csv", index=False)
    V3A_BEST = V3A_SELECTION
    best_model_a = V3A_MODELS[V3A_BEST["m"]]
    V3A_TEST, v3a_scores, v3a_y = v3_score_locked_test(best_model_a, V3A_OPTIONS[V3A_BEST["m"]],
                                                       v3_vocab, V3A_BEST["g"], V3A_BEST["r"], V3_BGL_COUNTS)
    print("V3 Phase A test verrouillé :")
    print(pd.DataFrame([V3A_TEST]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(pd.DataFrame([PAPER_BGL, {"version": f"V3 BGL m={V3A_BEST['m']:.2f},g={V3A_BEST['g']},r={V3A_BEST['r']:.2f}", **{k: V3A_TEST[k] for k in ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]}}]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    np.savez(WORK_DIR / "v3_bgl_phaseA.npz", y_test=v3a_y, scores=v3a_scores,
             g=np.array([V3A_BEST["g"]]), r=np.array([V3A_BEST["r"]]))
else:
    print("RUN_PHASE_A=False : sélection sautée.")""")

md("v3-multiseed-title", """## Phase B — Répétition multi-seed sur le `m` sélectionné

Splits figés (seed 12 + fractions fixes) : seules l'initialisation, l'entraînement et le masquage varient avec le seed. Pour chaque seed, `(g, r)` est re-sélectionné sur validation avec le protocole identique, puis le test verrouillé est scoré. Résultat : moyenne ± écart-type (n=3), sans toucher au test avant le gel par seed.""")

code("v3-multiseed", """V3B_ROWS, V3B_SCORES = [], {}
if RUN_PHASE_B:
    if "V3A_BEST" not in globals():
        raise RuntimeError("Phase B sans Phase A : relancer la sélection (RUN_PHASE_A=True) ou fixer V3A_BEST manuellement.")
    for seed in V3_SEEDS[1:]:
        model, options = v3_train_model(seed, V3A_BEST["m"])
        single_m, single_o = {V3A_BEST["m"]: model}, {V3A_BEST["m"]: options}
        sel, _ = v3_select_triplet(single_m, single_o, v3_vocab, V3_BGL_COUNTS,
                                   V3_G_GRID, V3_R_GRID, V3_TARGET_PREVALENCE)
        sel["seed"] = seed
        test_result, scores, y_test = v3_score_locked_test(model, options, v3_vocab,
                                                           sel["g"], sel["r"], V3_BGL_COUNTS)
        test_result["seed"] = seed
        test_result.update({f"sel_{k}": sel[k] for k in ["m", "g", "r", "validation_F1_target"]})
        V3B_ROWS.append(test_result)
        V3B_SCORES[seed] = {"scores": scores, "g": sel["g"], "r": sel["r"]}
        del model
    # Seed 1234 (Phase A) rejoint l'agrégat avec sa propre sélection.
    V3A_TEST["seed"] = V3_SEEDS[0]
    V3A_TEST.update({f"sel_{k}": V3A_SELECTION[k] for k in ["m", "g", "r", "validation_F1_target"]})
    V3B_SCORES[V3_SEEDS[0]] = {"scores": v3a_scores, "g": V3A_BEST["g"], "r": V3A_BEST["r"]}
    all_rows = [V3A_TEST] + V3B_ROWS
    summary = pd.DataFrame(all_rows)
    print(summary[["seed", "m", "g", "r", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "PR_AUC"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    agg = {}
    for metric in ["Precision", "Recall", "F1", "FPR", "ROC_AUC", "PR_AUC"]:
        values = summary[metric].to_numpy(dtype=float)
        agg[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 2 else 0.0,
                       "min": float(values.min()), "max": float(values.max())}
    print("Moyenne ± écart-type (n=%d) :" % len(summary))
    for metric, stats in agg.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}  [min {stats['min']:.4f}, max {stats['max']:.4f}]")
    (WORK_DIR / "logbert_v3_bgl_results.json").write_text(json.dumps(
        {"phaseA_selection": V3A_SELECTION, "phaseA_test": V3A_TEST, "phaseB_rows": all_rows,
         "aggregate_mean_std_n3": agg, "paper_reference": PAPER_BGL, "hdfs_v2_reference": HDFS_V2_REF,
         "seeds": V3_SEEDS, "m_grid": V3_M_GRID, "target_prevalence": V3_TARGET_PREVALENCE,
         "counts": V3_BGL_COUNTS, "repo_commit": REPO_COMMIT}, indent=2,
        default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
    np.savez(WORK_DIR / "v3_bgl_phaseB.npz", y_test=y_test,
             **{f"scores_seed{s}": V3B_SCORES[s]["scores"] for s in V3B_SCORES})
    print("Résultats V3 sauvés :", WORK_DIR / "logbert_v3_bgl_results.json")
else:
    print("RUN_PHASE_B=False : multi-seed sauté.")""")

md("v3-diag-title", """## Diagnostics V3 : confusion, ROC/PR par seed, comparaison croisée HDFS/BGL

Métriques au seuil gelé par seed. Aucune re-sélection sur le test.""")

code("v3-diag", """import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, average_precision_score

if ("V3B_SCORES" not in globals() or not V3B_SCORES):
    if not (WORK_DIR / "logbert_v3_bgl_results.json").exists() or not (WORK_DIR / "v3_bgl_phaseB.npz").exists():
        raise RuntimeError("Aucun résultat Phase A/B en mémoire ni sur disque : exécuter les phases dans l'ordre.")
    saved = json.loads((WORK_DIR / "logbert_v3_bgl_results.json").read_text())
    archive = np.load(WORK_DIR / "v3_bgl_phaseB.npz")
    y_test_v3 = archive["y_test"]
    V3B_SCORES = {int(r["seed"]): {"scores": archive[f"scores_seed{int(r['seed'])}"], "g": int(r["g"]), "r": float(r["r"])}
                  for r in saved["phaseB_rows"]}
    V3B_ROWS_FALLBACK = saved["phaseB_rows"]
    print("Scores rechargés depuis le disque (%d seeds)." % len(V3B_SCORES))
else:
    y_test_v3 = v3a_y
    V3B_ROWS_FALLBACK = [V3A_TEST] + V3B_ROWS

best_row = max(V3B_ROWS_FALLBACK, key=lambda r: r["F1"])
best_seed = int(best_row["seed"])
print(f"Meilleur seed test : {best_seed} (F1={best_row['F1']:.4f}, m={best_row['m']}, g={best_row['g']}, r={best_row['r']})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]
cm = confusion_matrix(y_test_v3, (V3B_SCORES[best_seed]["scores"] > best_row["r"]).astype(int), labels=[0, 1])
row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
image = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}\\n{row_pct[i,j]:.1%}", ha="center", va="center", fontsize=11,
                color="white" if cm[i,j] > cm.max() * .55 else "#1d2733")
ax.set_xticks([0, 1], ["Prédit normal", "Prédit anomalie"])
ax.set_yticks([0, 1], ["Réel normal", "Réel anomalie"])
ax.set_title(f"Confusion BGL seed {best_seed}")
fig.colorbar(image, ax=ax, fraction=.046, pad=.04)

ax = axes[1]
for seed, pack in sorted(V3B_SCORES.items()):
    fpr_c, tpr_c, _ = roc_curve(y_test_v3, pack["scores"])
    ax.plot(fpr_c, tpr_c, linewidth=2, label=f"seed {seed} AUC={auc(fpr_c, tpr_c):.3f}")
ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Hasard")
ax.set(xlabel="FPR", ylabel="Recall", title="ROC BGL par seed", xlim=(0, 1), ylim=(0, 1.02))
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.22)

ax = axes[2]
for seed, pack in sorted(V3B_SCORES.items()):
    pr_p, pr_r, _ = precision_recall_curve(y_test_v3, pack["scores"])
    ax.plot(pr_r, pr_p, linewidth=2, label=f"seed {seed} AP={average_precision_score(y_test_v3, pack['scores']):.3f}")
ax.axhline(float(y_test_v3.mean()), color="grey", linestyle=":", label=f"Prévalence={y_test_v3.mean():.2%}")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall BGL par seed", xlim=(0, 1), ylim=(0, 1.02))
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.22)
fig.suptitle("V3 BGL — diagnostics par seed (seuils gelés sur validation)")
fig.tight_layout(); fig.savefig(WORK_DIR / "logbert_v3_diagnostics.png", dpi=220, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ["Precision", "Recall", "F1"]
x = np.arange(3); width = .22
paper_vals = [PAPER_BGL[k] for k in labels]
hdfs_vals = [HDFS_V2_REF[k] for k in labels]
bgl_means = [np.mean([r[k] for r in V3B_ROWS_FALLBACK]) for k in labels]
bgl_stds = [np.std([r[k] for r in V3B_ROWS_FALLBACK], ddof=1) if len(V3B_ROWS_FALLBACK) > 2 else 0.0 for k in labels]
ax.bar(x - width, paper_vals, width, label="Article BGL", color="#5B7DB1")
ax.bar(x, hdfs_vals, width, label="V2 HDFS (1 seed)", color="#E39C43")
bars = ax.bar(x + width, bgl_means, width, yerr=bgl_stds, capsize=4, label="V3 BGL (moy. 3 seeds)", color="#4E9A75")
for bar, val in zip(bars, bgl_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .012, f"{val:.2f}", ha="center", fontsize=8)
ax.set_xticks(x, labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Papier BGL vs V2 HDFS vs V3 BGL — test verrouillé")
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.22)
fig.tight_layout(); fig.savefig(WORK_DIR / "logbert_v3_cross_dataset.png", dpi=220, bbox_inches="tight"); plt.show()""")

md("v3-bundle-title", """## Export du meilleur modèle BGL et API d'inférence

Même contrat que V2 : checkpoint + vocabulaire + mapping Drain + `inference.py` générique (fonction de scoring/preuves pour le SLM). La carte des templates vient de `BGL.log_templates.csv` produit par Drain. Smoke test sur validation uniquement.""")

bundle_code = '''import os, json, shutil, zipfile, textwrap
from pathlib import Path

for _need in ["v3_train_model", "v3_vocab", "v3_vocab_path"]:
    if _need not in globals():
        raise RuntimeError("Exécuter d'abord les cellules Dépendances, Parsing et Phase A (même avec RUN_PHASE_A=False).")
if "V3B_ROWS_FALLBACK" not in globals():
    saved_bundle = json.loads((WORK_DIR / "logbert_v3_bgl_results.json").read_text())
    V3B_ROWS_FALLBACK = saved_bundle["phaseB_rows"]
best_bundle_row = max(V3B_ROWS_FALLBACK, key=lambda r: r["F1"])
BEST_SEED_BGL = int(best_bundle_row["seed"])
BEST_M_BGL, BEST_G_BGL, BEST_R_BGL = float(best_bundle_row["m"]), int(best_bundle_row["g"]), float(best_bundle_row["r"])
print(f"Export seed {BEST_SEED_BGL} : m={BEST_M_BGL}, g={BEST_G_BGL}, r={BEST_R_BGL}, F1 test={best_bundle_row['F1']:.4f}")

ARTIFACT_BGL = Path("/kaggle/working/logbert_v3_bgl_best")
if ARTIFACT_BGL.exists():
    shutil.rmtree(ARTIFACT_BGL)
ARTIFACT_BGL.mkdir(parents=True, exist_ok=True)

best_bgl_model, best_bgl_options = v3_train_model(BEST_SEED_BGL, BEST_M_BGL)
best_checkpoint_bgl = Path(best_bgl_options["model_path"])
shutil.copy2(best_checkpoint_bgl, ARTIFACT_BGL / "best_bert.pth")
shutil.copy2(v3_vocab_path, ARTIFACT_BGL / "vocab.pkl")
bgl_templates_csv = OUTPUT_BGL / "BGL.log_templates.csv"
if bgl_templates_csv.exists():
    shutil.copy2(bgl_templates_csv, ARTIFACT_BGL / "BGL.log_templates.csv")
    template_frame = pd.read_csv(bgl_templates_csv, usecols=["EventId", "EventTemplate"]).drop_duplicates("EventId")
    id_to_template_bgl = dict(zip(template_frame.EventId.astype(str), template_frame.EventTemplate.astype(str)))
else:
    id_to_template_bgl = {}
    print("Attention : BGL.log_templates.csv absent, carte des templates vide.")
event_hash_to_id_bgl = {eid: eid for eid in id_to_template_bgl}

manifest_bgl = {
    "model": "Official LogBERT BGL implementation",
    "repository": "https://github.com/HelenGuohx/LogBERT",
    "repository_commit": REPO_COMMIT,
    "dataset": "LogHub BGL full (5-min windows, 1-min step)",
    "artifact_type": "serialized full PyTorch model plus WordVocab pickle",
    "mask_ratio_m": BEST_M_BGL,
    "top_g": BEST_G_BGL,
    "sequence_threshold_r": BEST_R_BGL,
    "sequence_length": int(best_bgl_options["seq_len"]),
    "adaptive_window": bool(best_bgl_options["adaptive_window"]),
    "minimum_sequence_length": int(best_bgl_options["min_len"]),
    "prediction_rule": "anomaly iff undetected masked keys > masked keys * r",
    "input_contract": "list of per-session BGL EventId strings from the bundled Drain mapping",
    "id_to_template": id_to_template_bgl,
    "event_hash_to_id": {str(k): str(v) for k, v in event_hash_to_id_bgl.items()},
    "test_metrics": {k: v for k, v in best_bundle_row.items() if k in ["Precision", "Recall", "F1", "FPR", "TP", "FP", "FN", "TN", "ROC_AUC", "PR_AUC", "test_prevalence", "seed"]},
}
(ARTIFACT_BGL / "manifest.json").write_text(json.dumps(manifest_bgl, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
torch.save(best_bgl_model.state_dict(), ARTIFACT_BGL / "best_bert_state_dict.pth")

''' + INFERENCE_BLOCK + '''(ARTIFACT_BGL / "inference.py").write_text(inference_source, encoding="utf-8")
(ARTIFACT_BGL / "README_inference.txt").write_text(
    "V3 best BGL LogBERT inference artifact. Input is a per-session list of BGL EventId strings from the included official Drain mapping. "
    "Add the official LogBERT repository at the commit in manifest.json to PYTHONPATH before loading the serialized model. "
    "The inference function returns anomaly decision, score, and top-g masked-key evidence for downstream SLM explanation.\\n",
    encoding="utf-8",
)

bundle_zip_bgl = shutil.make_archive("/kaggle/working/logbert_v3_bgl_best", "zip", ARTIFACT_BGL)
print("Artifact directory:", ARTIFACT_BGL)
print("Downloadable bundle:", bundle_zip_bgl)

# Smoke test on BGL validation samples only; thresholds and test metrics untouched.
sys.path.insert(0, str(REPO_DIR))
inference_spec_bgl = importlib.util.spec_from_file_location("logbert_v3_inference", ARTIFACT_BGL / "inference.py")
inference_module_bgl = importlib.util.module_from_spec(inference_spec_bgl)
inference_spec_bgl.loader.exec_module(inference_module_bgl)
bundle_bgl = inference_module_bgl.load_logbert_bundle(ARTIFACT_BGL, repo_commit=REPO_COMMIT)
def v3_read_sequences(filename, n=1):
    with open(OUTPUT_BGL / filename, encoding="utf-8") as f:
        return [line.split()[:512] for _, line in zip(range(n), f)]
smoke_bgl = inference_module_bgl.predict_event_sequences(
    v3_read_sequences("val_normal") + v3_read_sequences("val_abnormal"),
    bundle_bgl, session_ids=["smoke-bgl-normal", "smoke-bgl-anomaly"], seed=BASE_SEED)
print("BGL inference smoke test:")
print(json.dumps(smoke_bgl, ensure_ascii=False, indent=2)[:4000])'''
code("v3-bundle", bundle_code)

md("v3-phasec-title", """## Phase C — Pipeline HDFS complet + seeds supplémentaires (autonome)

Phase C autonome : données HDFS (téléchargement si besoin), parsing Drain officiel, split V2 à l'identique, vocabulaire train-only, puis entraînement `m=0.75` sur les seeds 2026/999 (le seed 1234 de référence vient du run V2 : 0.8267/0.8362/0.8314). Sélection `(g, r)` par seed sur validation, test verrouillé, moyenne ± écart-type sur 3 seeds. Les étapes C-1 à C-3 sont coûteuses mais uniques (checkpoints repris ensuite).""")

md("v3-phasec-data-title", """### Phase C-1. Données HDFS

Même recherche que V2 : variables `HDFS_LOG_PATH`/`HDFS_LABEL_PATH`, dossier `/kaggle/input`, sinon archive LogHub `HDFS_v1.zip` (187 Mo, MD5 vérifié) + labels LogPai. Environ 1,5 Go extrait : vérifier l'espace disque.""")

code("v3-phasec-data", """explicit_log = os.environ.get("HDFS_LOG_PATH")
explicit_labels = os.environ.get("HDFS_LABEL_PATH")
if explicit_log and explicit_labels:
    HDFS_LOG, LABEL_CSV = Path(explicit_log).expanduser().resolve(), Path(explicit_labels).expanduser().resolve()
else:
    roots = [Path("/kaggle/input"), Path("/content"), Path.cwd()]
    if os.environ.get("HDFS_INPUT_DIR"):
        roots.insert(0, Path(os.environ["HDFS_INPUT_DIR"]).expanduser())
    roots = list(dict.fromkeys(root.resolve() for root in roots if root.exists()))
    log_candidates = []
    label_candidates = []
    for root in roots:
        log_candidates.extend(p for p in root.rglob("*") if p.is_file() and p.name.lower() == "hdfs.log" and p.stat().st_size >= 100_000_000)
        label_candidates.extend(p for p in root.rglob("*") if p.is_file() and p.name.lower() == "anomaly_label.csv")
    log_candidates = list(dict.fromkeys(log_candidates))
    label_candidates = list(dict.fromkeys(label_candidates))
    pairs = [(log_path, label_path) for log_path in log_candidates for label_path in label_candidates]
    if pairs:
        def common_depth(pair):
            return len(Path(os.path.commonpath([str(pair[0]), str(pair[1])])).parts)
        best_depth = max(common_depth(pair) for pair in pairs)
        best_pairs = [pair for pair in pairs if common_depth(pair) == best_depth]
    else:
        best_pairs = []
    if len(best_pairs) == 0:
        import hashlib, urllib.request, zipfile
        download_dir = WORK_DIR / "downloaded_hdfs_v1"
        download_dir.mkdir(parents=True, exist_ok=True)
        archive_path = download_dir / "HDFS_v1.zip"
        hdfs_target = download_dir / "HDFS.log"
        labels_target = download_dir / "anomaly_label.csv"
        archive_url = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
        expected_md5 = "76a24b4d9a6164d543fb275f89773260"
        label_url = "https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/anomaly_label.csv"
        if not hdfs_target.exists() or not labels_target.exists():
            print("HDFS complet non attache ; telechargement LogHub officiel (187 Mo, reprises auto)...")
            import time as _time
            _urls = [archive_url,
                     "https://zenodo.org/api/records/8196385/files/HDFS_v1.zip/content"]
            _ok, _last_error = False, None
            for _attempt in range(1, 5):
                for _url in _urls:
                    try:
                        print(f"Tentative {_attempt} : {_url[:70]}...")
                        _req = urllib.request.Request(_url, headers={"User-Agent": "Mozilla/5.0"})
                        _have = archive_path.stat().st_size if archive_path.exists() else 0
                        if _have:
                            _req.add_header("Range", f"bytes={_have}-")
                        with urllib.request.urlopen(_req, timeout=300) as _resp, \
                                archive_path.open("ab" if (_have and _resp.status == 206) else "wb") as _out:
                            while True:
                                _chunk = _resp.read(4 * 1024 * 1024)
                                if not _chunk:
                                    break
                                _out.write(_chunk)
                        if archive_path.stat().st_size == 186645559:
                            _ok = True
                            break
                        print(f"Archive incomplete ({archive_path.stat().st_size} octets), nouvel essai...")
                    except Exception as _e:
                        _last_error = _e
                        print(f"Echec tentative {_attempt} : {type(_e).__name__} {str(_e)[:150]}")
                if _ok:
                    break
                _time.sleep(30 * _attempt)
            if not _ok:
                raise RuntimeError(
                    "Telechargement HDFS impossible apres 6 tentatives (Zenodo 504). "
                    "Solutions : relancer la cellule, ou attacher HDFS.log (522 Mo+ decompresse : 1,5 Go) + "
                    "anomaly_label.csv via Add Input (dataset Kaggle 'hdfs'), "
                    "ou definir HDFS_LOG_PATH et HDFS_LABEL_PATH.") from _last_error
            _digest_final = hashlib.md5()
            with archive_path.open("rb") as _f:
                while True:
                    _chunk = _f.read(8 * 1024 * 1024)
                    if not _chunk:
                        break
                    _digest_final.update(_chunk)
            if _digest_final.hexdigest() != expected_md5:
                raise IOError(f"Checksum HDFS_v1.zip invalide : {_digest_final.hexdigest()} (fichier conservé pour reprise).")
            if not hdfs_target.exists():
                print("Extraction HDFS.log (1,5 Go)...")
                with zipfile.ZipFile(archive_path) as _archive:
                    _member = next((name for name in _archive.namelist() if Path(name).name == "HDFS.log"), None)
                    if _member is None:
                        raise FileNotFoundError("HDFS.log absent de larchive.")
                    with _archive.open(_member) as _src, hdfs_target.open("wb") as _dst:
                        shutil.copyfileobj(_src, _dst, length=8 * 1024 * 1024)
            if not labels_target.exists():
                for _ltry in range(1, 4):
                    try:
                        print(f"Labels LogPai (tentative {_ltry})...")
                        urllib.request.urlretrieve(label_url, labels_target)
                        break
                    except Exception as _le:
                        print(f"Labels : {type(_le).__name__} {str(_le)[:120]}")
                        _time.sleep(15 * _ltry)
                else:
                    raise RuntimeError("Labels introuvables : attacher anomaly_label.csv via Add Input ou HDFS_LABEL_PATH.")
        HDFS_LOG, LABEL_CSV = hdfs_target, labels_target
        print("HDFS officiel téléchargé:", HDFS_LOG, f"({HDFS_LOG.stat().st_size/1e9:.2f} GB)")
    elif len(best_pairs) > 1:
        raise FileNotFoundError(f"Plusieurs couples HDFS candidats ; fixer HDFS_LOG_PATH et HDFS_LABEL_PATH. {best_pairs[:10]}")
    else:
        HDFS_LOG, LABEL_CSV = best_pairs[0]
if not HDFS_LOG.is_file() or not LABEL_CSV.is_file():
    raise FileNotFoundError(f"Fichiers HDFS introuvables: {HDFS_LOG}, {LABEL_CSV}")
if HDFS_LOG.stat().st_size < 100_000_000:
    raise ValueError("HDFS.log incomplet (extrait ?).")
label_df = pd.read_csv(LABEL_CSV)
label_df["BlockId"] = label_df.BlockId.astype(str)
print("HDFS.log:", HDFS_LOG, f"({HDFS_LOG.stat().st_size/1e9:.2f} GB)")
print("Labels:", len(label_df), label_df.Label.value_counts().to_dict())
assert {"BlockId", "Label"}.issubset(label_df.columns)
assert label_df.BlockId.is_unique
assert set(label_df.Label.unique()).issubset({"Normal", "Anomaly"})""")

md("v3-phasec-parse-title", """### Phase C-2. Parsing Drain officiel HDFS

`dp.parser` (Drain depth=5, st=0.5), `dp.mapping`, `dp.hdfs_sampling` : exactement le pipeline V2, dans un clone séparé (`output/hdfs`).""")

code("v3-phasec-parse", """REPO_HDFS = REPO_DIR / "HDFS"
OUTPUT_HDFS = REPO_DIR / "output" / "hdfs"
OUTPUT_HDFS.mkdir(parents=True, exist_ok=True)
(OUTPUT_HDFS / "bert").mkdir(parents=True, exist_ok=True)

DATA_HOME_HDFS = Path.home() / ".dataset" / "hdfs"
DATA_HOME_HDFS.mkdir(parents=True, exist_ok=True)
for source, target in [(HDFS_LOG, DATA_HOME_HDFS / "HDFS.log"), (LABEL_CSV, DATA_HOME_HDFS / "anomaly_label.csv")]:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)

spec_hdfs = importlib.util.spec_from_file_location("hdfs_data_process", REPO_HDFS / "data_process.py")
dp_hdfs = importlib.util.module_from_spec(spec_hdfs)
spec_hdfs.loader.exec_module(dp_hdfs)

old_cwd = Path.cwd()
os.chdir(REPO_HDFS)
try:
    log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
    dp_hdfs.parser(str(DATA_HOME_HDFS) + "/", "../output/hdfs/", "HDFS.log", log_format, "drain")
    dp_hdfs.mapping()
    dp_hdfs.hdfs_sampling("../output/hdfs/HDFS.log_structured.csv")
finally:
    os.chdir(old_cwd)

STRUCTURED_HDFS = OUTPUT_HDFS / "HDFS.log_structured.csv"
SEQUENCE_HDFS = OUTPUT_HDFS / "hdfs_sequence.csv"
assert STRUCTURED_HDFS.exists() and SEQUENCE_HDFS.exists(), "Parsing HDFS sans sorties."
print("Structured:", list(pd.read_csv(STRUCTURED_HDFS, nrows=5).columns))""")

md("v3-phasec-split-title", """### Phase C-3. Split V2 à l'identique

Shuffle seed 20, train 4 855 normales, validation 10 % / 20 %, `min_len=10`, disjonction `BlockId` vérifiée entre les 5 partitions.""")

code("v3-phasec-split", """sequence_df = pd.read_csv(SEQUENCE_HDFS, dtype={"BlockId": str})
sequence_df["events"] = sequence_df.EventSequence.map(ast.literal_eval)
label_map = dict(zip(label_df.BlockId, label_df.Label))
sequence_df["label"] = sequence_df.BlockId.map(label_map)
sequence_df = sequence_df.dropna(subset=["label"]).reset_index(drop=True)
sequence_df["y"] = (sequence_df.label == "Anomaly").astype(int)
print("Sequences sans label:", 0)

normal_all_hdfs = sequence_df[sequence_df.y == 0].sample(frac=1, random_state=20).reset_index(drop=True)
anomaly_all_hdfs = sequence_df[sequence_df.y == 1].sample(frac=1, random_state=20).reset_index(drop=True)
N_TRAIN_NORMAL = 4855
if len(normal_all_hdfs) <= N_TRAIN_NORMAL:
    raise ValueError("HDFS complet requis.")
hdfs_train_df = normal_all_hdfs.iloc[:N_TRAIN_NORMAL].copy()
remaining_hdfs = normal_all_hdfs.iloc[N_TRAIN_NORMAL:].reset_index(drop=True)
n_val_n_h = int(len(remaining_hdfs) * 0.10)
n_val_a_h = int(len(anomaly_all_hdfs) * 0.20)
hdfs_val_normal = remaining_hdfs.iloc[:n_val_n_h].copy()
hdfs_test_normal = remaining_hdfs.iloc[n_val_n_h:].copy()
hdfs_val_anomaly = anomaly_all_hdfs.iloc[:n_val_a_h].copy()
hdfs_test_anomaly = anomaly_all_hdfs.iloc[n_val_a_h:].copy()

MIN_LEN = 10
def keep_eligible_hdfs(frame):
    return frame[frame.events.map(len) >= MIN_LEN].reset_index(drop=True)
hdfs_train_df, hdfs_val_normal, hdfs_test_normal = map(keep_eligible_hdfs, [hdfs_train_df, hdfs_val_normal, hdfs_test_normal])
hdfs_val_anomaly, hdfs_test_anomaly = map(keep_eligible_hdfs, [hdfs_val_anomaly, hdfs_test_anomaly])
assert hdfs_train_df.y.eq(0).all()
_all_ids_h = [set(x.BlockId) for x in [hdfs_train_df, hdfs_val_normal, hdfs_val_anomaly, hdfs_test_normal, hdfs_test_anomaly]]
for i in range(len(_all_ids_h)):
    for j in range(i + 1, len(_all_ids_h)):
        assert _all_ids_h[i].isdisjoint(_all_ids_h[j]), "BlockId overlap."

def write_hdfs_sequences(frame, filename):
    path = OUTPUT_HDFS / filename
    with path.open("w", encoding="utf-8") as stream:
        for events in frame.events:
            stream.write(" ".join(map(str, events)) + "\\n")
    return path

for frame, name in [(hdfs_train_df, "train"), (hdfs_val_normal, "val_normal"), (hdfs_val_anomaly, "val_abnormal"),
                    (hdfs_test_normal, "test_normal"), (hdfs_test_anomaly, "test_abnormal")]:
    write_hdfs_sequences(frame, name)

HDFS_COUNTS = {"val_normal": len(hdfs_val_normal), "val_abnormal": len(hdfs_val_anomaly),
               "test_normal": len(hdfs_test_normal), "test_abnormal": len(hdfs_test_anomaly)}
print(f"Train={len(hdfs_train_df):,} ; val {HDFS_COUNTS['val_normal']:,}/{HDFS_COUNTS['val_abnormal']:,} ; "
      f"test {HDFS_COUNTS['test_normal']:,}/{HDFS_COUNTS['test_abnormal']:,}")
print("Prévalence test HDFS: %.3f%%" % (100 * HDFS_COUNTS["test_abnormal"] / (HDFS_COUNTS["test_normal"] + HDFS_COUNTS["test_abnormal"])))
(WORK_DIR / "v3_hdfs_split.json").write_text(json.dumps({"counts": HDFS_COUNTS, "repo_commit": REPO_COMMIT}, indent=2))""")

md("v3-phasec-train-title", """### Phase C-4. Entraînements HDFS `m=0.75` (seeds 2026/999)

Config officielle `HDFS/logbert.py` (seul `mask_ratio` est forcé à 0.75), vocabulaire train-only, reprise sur checkpoint existant.""")

code("v3-phasec-train", """for _need in ["Trainer", "seed_everything", "WordVocab"]:
    if _need not in globals():
        raise RuntimeError("Exécuter d'abord la cellule d'entraînement Phase A (imports partagés).")
subprocess.run([sys.executable, "logbert.py", "vocab"], cwd=REPO_HDFS, check=True)
hdfs_vocab = WordVocab.load_vocab(str(OUTPUT_HDFS / "vocab.pkl"))
print("Vocabulaire HDFS (train only) :", len(hdfs_vocab))
spec_hdfs_c = importlib.util.spec_from_file_location("hdfs_logbert_phasec_config", REPO_HDFS / "logbert.py")
cfg_hdfs = importlib.util.module_from_spec(spec_hdfs_c)
spec_hdfs_c.loader.exec_module(cfg_hdfs)
HDFS_G_GRID = [g for g in V3_G_GRID_FULL if g <= len(hdfs_vocab)]
print("Grille g HDFS :", HDFS_G_GRID)
HDFS_M = 0.75
HDFS_EXTRA_SEEDS = [2026, 999]

def hdfs_train_model(seed):
    model_dir = OUTPUT_HDFS / "v3_hdfs" / f"seed{seed}" / "m0p75"
    model_dir.mkdir(parents=True, exist_ok=True)
    options = cfg_hdfs.options.copy()
    options.update({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": str(OUTPUT_HDFS) + "/",
        "model_dir": str(model_dir) + "/",
        "model_path": str(model_dir / "best_bert.pth"),
        "vocab_path": str(OUTPUT_HDFS / "vocab.pkl"),
        "scale_path": str(model_dir / "scale.pkl"),
        "mask_ratio": HDFS_M,
        "num_workers": 2,
        "on_memory": True,
    })
    seed_everything(seed=seed)
    checkpoint = Path(options["model_path"])
    if checkpoint.exists():
        print(f"HDFS checkpoint seed={seed} m=0.75 : reprise sans réentraînement.")
    else:
        print(f"HDFS training seed={seed} m=0.75")
        trainer_h = Trainer(options)
        trainer_h.train()
        del trainer_h
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint HDFS manquant seed={seed}")
    try:
        model = torch.load(str(checkpoint), map_location=options["device"], weights_only=False)
    except TypeError:
        model = torch.load(str(checkpoint), map_location=options["device"])
    model.to(options["device"]); model.eval()
    return model, options

HDFS_OPTIONS = {}
if RUN_PHASE_C:
    for _seed in HDFS_EXTRA_SEEDS:
        _, HDFS_OPTIONS[_seed] = hdfs_train_model(_seed)
    print("Phase C entraînements OK :", sorted(HDFS_OPTIONS))
else:
    print("RUN_PHASE_C=False : entraînements HDFS sautés.")""")

md("v3-phasec-select-title", """### Phase C-5. Sélection par seed + test verrouillé + agrégat 3 seeds

Protocole identique (prior HDFS, mêmes grilles) via les fonctions Phase A. La ligne seed 1234 est la référence V2 figée (run `logbert_hdfs_v2_resultats.ipynb`).""")

code("v3-phasec-select", """for _need in ["v3_select_triplet", "v3_score_locked_test"]:
    if _need not in globals():
        raise RuntimeError("Exécuter d'abord la cellule de sélection Phase A (fonctions partagées).")
HDFS_PRIOR = 10647 / (553366 + 10647)
HDFS_REF1234 = {"seed": 1234, "m": 0.75, "g": 8, "r": 0.04,
                "validation_F1_target": 0.8493, "validation_precision_target": 0.8677, "validation_recall": 0.8317,
                "Precision": 0.8267, "Recall": 0.8362, "F1": 0.8314, "FPR": 1491 / 498032,
                "TP": 7114, "FP": 1491, "FN": 1394, "TN": 496541,
                "ROC_AUC": 0.9235, "PR_AUC": 0.7051, "test_prevalence": 8508 / 506540,
                "Precision_CI95": [0.8186, 0.8346], "Recall_CI95": [0.8281, 0.8439]}
hdfs_rows = [dict(HDFS_REF1234)]
_npz_hdfs = {}
if RUN_PHASE_C:
    for _seed in HDFS_EXTRA_SEEDS:
        _model, _ = hdfs_train_model(_seed)
        _sel, _ = v3_select_triplet({HDFS_M: _model}, {HDFS_M: HDFS_OPTIONS[_seed]}, hdfs_vocab,
                                    HDFS_COUNTS, HDFS_G_GRID, V3_R_GRID, HDFS_PRIOR)
        _test, _scores, _y = v3_score_locked_test(_model, HDFS_OPTIONS[_seed], hdfs_vocab,
                                                  _sel["g"], _sel["r"], HDFS_COUNTS)
        _row = {"seed": _seed}
        _row.update({k: _sel[k] for k in ["m", "g", "r", "validation_F1_target", "validation_precision_target", "validation_recall"]})
        _row.update({k: _test[k] for k in ["Precision", "Recall", "F1", "FPR", "TP", "FP", "FN", "TN", "ROC_AUC", "PR_AUC", "test_prevalence", "Precision_CI95", "Recall_CI95"]})
        hdfs_rows.append(_row)
        _npz_hdfs[f"scores_seed{_seed}"] = _scores
        _y_hdfs_last = _y
        del _model
    print(pd.DataFrame(hdfs_rows)[["seed", "m", "g", "r", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "PR_AUC"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    _agg_h = {}
    _frame_h = pd.DataFrame(hdfs_rows)
    for _metric in ["Precision", "Recall", "F1", "FPR", "ROC_AUC", "PR_AUC"]:
        _vals = _frame_h[_metric].to_numpy(dtype=float)
        _agg_h[_metric] = {"mean": float(_vals.mean()), "std": float(_vals.std(ddof=1)) if len(_vals) > 2 else 0.0,
                           "min": float(_vals.min()), "max": float(_vals.max())}
    print("HDFS moyenne ± écart-type (n=%d) :" % len(_frame_h))
    for _metric, _stats in _agg_h.items():
        print(f"  {_metric}: {_stats['mean']:.4f} ± {_stats['std']:.4f}  [min {_stats['min']:.4f}, max {_stats['max']:.4f}]")
    (WORK_DIR / "logbert_v3_hdfs_extraseeds.json").write_text(json.dumps(
        {"rows": hdfs_rows, "aggregate_mean_std_n3": _agg_h, "repo_commit": REPO_COMMIT}, indent=2,
        default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
    np.savez(WORK_DIR / "v3_hdfs_phaseC.npz", y_test=_y_hdfs_last, **_npz_hdfs)
    print("Phase C sauvée :", WORK_DIR / "logbert_v3_hdfs_extraseeds.json")
else:
    print("RUN_PHASE_C=False : sélection HDFS sautée.")""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10.0"}},
      "cells": cells}
json.dump(nb, open("logbert_v3_bgl.ipynb", "w"), ensure_ascii=False, indent=1)
print("Notebook V3 écrit :", len(cells), "cellules.")

# Syntax check (strip Kaggle magics).
import py_compile
bad = 0
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "\n".join(l for l in "".join(c["source"]).splitlines()
                    if not l.lstrip().startswith(("%", "!")))
    try:
        compile(src, f"cell-{i}-{c['id']}", "exec")
    except SyntaxError as e:
        bad += 1
        print(f"SYNTAX cell {i} ({c['id']}): {e}")
print("Syntaxe :", "OK" if bad == 0 else f"{bad} cellules en erreur")
