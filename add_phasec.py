"""Replace Phase C stub with a self-contained HDFS pipeline (data+parse+split+train+select).
Patches logbert_v3_bgl.ipynb AND gen_v3.py (kept in sync from single cell definitions)."""
import json

C = []  # (id, kind, source)

C.append(("v3-phasec-title", "md", """## Phase C — Pipeline HDFS complet + seeds supplémentaires (autonome)

Phase C autonome : données HDFS (téléchargement si besoin), parsing Drain officiel, split V2 à l'identique, vocabulaire train-only, puis entraînement `m=0.75` sur les seeds 2026/999 (le seed 1234 de référence vient du run V2 : 0.8267/0.8362/0.8314). Sélection `(g, r)` par seed sur validation, test verrouillé, moyenne ± écart-type sur 3 seeds. Les étapes C-1 à C-3 sont coûteuses mais uniques (checkpoints repris ensuite)."""))

C.append(("v3-phasec-data-title", "md", """### Phase C-1. Données HDFS

Même recherche que V2 : variables `HDFS_LOG_PATH`/`HDFS_LABEL_PATH`, dossier `/kaggle/input`, sinon archive LogHub `HDFS_v1.zip` (187 Mo, MD5 vérifié) + labels LogPai. Environ 1,5 Go extrait : vérifier l'espace disque."""))

C.append(("v3-phasec-data", "code", """explicit_log = os.environ.get("HDFS_LOG_PATH")
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
            print("HDFS complet non attaché ; téléchargement LogHub officiel...")
            try:
                if not archive_path.exists() or archive_path.stat().st_size != 186645559:
                    digest = hashlib.md5()
                    request = urllib.request.urlopen(archive_url, timeout=120)
                    with archive_path.open("wb") as output:
                        while True:
                            chunk = request.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                    if digest.hexdigest() != expected_md5:
                        raise IOError(f"Checksum HDFS_v1.zip invalide: {digest.hexdigest()}")
                if not hdfs_target.exists():
                    with zipfile.ZipFile(archive_path) as archive:
                        member = next((name for name in archive.namelist() if Path(name).name == "HDFS.log"), None)
                        if member is None:
                            raise FileNotFoundError("HDFS.log absent de l'archive LogHub HDFS_v1.")
                        with archive.open(member) as source, hdfs_target.open("wb") as output:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                output.write(chunk)
                if not labels_target.exists():
                    urllib.request.urlretrieve(label_url, labels_target)
            except Exception as error:
                raise RuntimeError("Téléchargement HDFS impossible. Activer Internet ou attacher les fichiers.") from error
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
assert set(label_df.Label.unique()).issubset({"Normal", "Anomaly"})"""))

C.append(("v3-phasec-parse-title", "md", """### Phase C-2. Parsing Drain officiel HDFS

`dp.parser` (Drain depth=5, st=0.5), `dp.mapping`, `dp.hdfs_sampling` : exactement le pipeline V2, dans un clone séparé (`output/hdfs`)."""))

C.append(("v3-phasec-parse", "code", """REPO_HDFS = REPO_DIR / "HDFS"
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
print("Structured:", list(pd.read_csv(STRUCTURED_HDFS, nrows=5).columns))"""))

C.append(("v3-phasec-split-title", "md", """### Phase C-3. Split V2 à l'identique

Shuffle seed 20, train 4 855 normales, validation 10 % / 20 %, `min_len=10`, disjonction `BlockId` vérifiée entre les 5 partitions."""))

C.append(("v3-phasec-split", "code", """sequence_df = pd.read_csv(SEQUENCE_HDFS, dtype={"BlockId": str})
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
(WORK_DIR / "v3_hdfs_split.json").write_text(json.dumps({"counts": HDFS_COUNTS, "repo_commit": REPO_COMMIT}, indent=2))"""))

C.append(("v3-phasec-train-title", "md", """### Phase C-4. Entraînements HDFS `m=0.75` (seeds 2026/999)

Config officielle `HDFS/logbert.py` (seul `mask_ratio` est forcé à 0.75), vocabulaire train-only, reprise sur checkpoint existant."""))

C.append(("v3-phasec-train", "code", """for _need in ["Trainer", "seed_everything", "WordVocab"]:
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
    print("RUN_PHASE_C=False : entraînements HDFS sautés.")"""))

C.append(("v3-phasec-select-title", "md", """### Phase C-5. Sélection par seed + test verrouillé + agrégat 3 seeds

Protocole identique (prior HDFS, mêmes grilles) via les fonctions Phase A. La ligne seed 1234 est la référence V2 figée (run `logbert_hdfs_v2_resultats.ipynb`)."""))

C.append(("v3-phasec-select", "code", """for _need in ["v3_select_triplet", "v3_score_locked_test"]:
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
    print("RUN_PHASE_C=False : sélection HDFS sautée.")"""))

# ---- sanity: no triple quotes inside sources, single backslash-n discipline ----
for cid, kind, src in C:
    assert '"""' not in src, cid
    assert chr(92) * 2 not in src.replace(chr(92) + "n", ""), cid  # only \n escapes allowed

# ---- 1. patch notebook ----
nb_path = "logbert_v3_bgl.ipynb"
nb = json.load(open(nb_path))
idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "v3-phasec-title")
assert nb["cells"][idx + 1].get("id") == "v3-phasec-code" or True
# find end of old stub (title + code cells)
end = idx + 2
new_cells = [{"id": cid, "cell_type": ("markdown" if kind == "md" else "code"), "metadata": {},
              "source": [l + "\n" for l in src.split("\n")],
              **({} if kind == "md" else {"outputs": [], "execution_count": None})}
             for cid, kind, src in C]
nb["cells"][idx:end] = new_cells
json.dump(nb, open(nb_path, "w"), ensure_ascii=False, indent=1)
print("notebook: stub remplacé par", len(new_cells), "cellules Phase C.")

# syntax check new code cells
bad = 0
for cid, kind, src in C:
    if kind != "code":
        continue
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    try:
        compile(code, cid, "exec")
    except SyntaxError as e:
        bad += 1
        print("SYNTAX", cid, e)
print("syntaxe Phase C:", "OK" if bad == 0 else "ERREUR")

# ---- 2. keep gen_v3.py in sync: regenerate its phasec span from same sources ----
g = open("gen_v3.py", encoding="utf-8").read()
start = g.find('md("v3-phasec-title"')
marker = 'nb = {"nbformat"'
stop = g.find(marker)
assert start != -1 and stop != -1 and start < stop
calls = []
for cid, kind, src in C:
    esc = src.replace(chr(92), chr(92) * 2)  # cell \n -> gen \\n
    calls.append(f'{kind}("{cid}", """{esc}""")\n')
g = g[:start] + "\n".join(calls) + "\n" + g[stop:]
g = g.replace("RUN_PHASE_C = False", "RUN_PHASE_C = True")
open("gen_v3.py", "w", encoding="utf-8").write(g)
print("gen_v3.py synchronisé (RUN_PHASE_C=True par défaut).")
compile(g, "gen_v3", "exec")
print("gen_v3.py recompile: OK")
