"""
Builds the five Phase 4 Colab training notebooks
(train_m3.ipynb .. train_m2.ipynb) into training/notebooks/.

Adapted from MENTORA_Phase4_Model_Training.md to match the ACTUAL schemas
produced by Phase 3's scripts (see datasets/SOURCES.md):
  - M4's training_pairs.csv columns are (required_skills, job_title, label),
    not (skill_text, career_text, label) — required_skills is a stringified
    list, joined into a skill_text string here.
  - M5 currently only has a single gold_val_test.jsonl (no separate silver
    set yet, since the starter resumes are all hand-authored/"gold" per
    SOURCES.md) — the M5 notebook splits gold into train/val/test instead
    of the silver/gold split the doc describes, with a clear TODO for once
    real Kaggle resumes + a silver set exist.
  - M1's `topic` column is single-label per question (not '|'-separated
    multi-topic) — the multihot encoding still works unchanged since
    `"Algebra".split('|')` just returns `["Algebra"]`.

Run: python3 build_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

OUT_DIR = Path(__file__).parent / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU",
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


SHARED_SETUP = [
    md("## 0. Shared setup — mount Drive, confirm GPU, install deps\n"
       "Run these three cells first in every notebook (per Phase 4 §0)."),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n\n"
        "import subprocess\n"
        "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n"
        "# Expect to see \"Tesla T4\" — if this errors or shows no GPU, go to\n"
        "# Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU -> Save, then re-run."
    ),
    code(
        "import os\n"
        "DATA = '/content/drive/MyDrive/mentora_data'\n"
        "MODELS = '/content/drive/MyDrive/mentora_models'\n"
        "for m in ['model1_gap_classifier', 'model2_question_generator',\n"
        "          'model3_trajectory_predictor', 'model4_career_matcher', 'model5_cv_ner']:\n"
        "    os.makedirs(f'{MODELS}/{m}', exist_ok=True)\n\n"
        "# NOTE: upload the repo's datasets/ folder to Google Drive at this path first\n"
        "# (mentora_data/{raw,processed,labeled}/...) — see datasets/README.md."
    ),
]


def pip_cell(extra=""):
    return code(
        "!pip install -q torch --index-url https://download.pytorch.org/whl/cu121  "
        "# Colab GPU wheel — do NOT reuse the CPU wheel from the backend's laptop plan\n"
        f"!pip install -q transformers peft accelerate sentence-transformers datasets evaluate seqeval spacy sacrebleu rouge_score{extra}"
    )


WANDB_CELL_MD = md(
    "### Optional — Weights & Biases logging\n"
    "Nice for live loss curves / a thesis screenshot. Skip entirely if you'd rather "
    "keep things simple — nothing below strictly needs it."
)
WANDB_CELL_CODE = code(
    "# !pip install -q wandb\n"
    "# import wandb\n"
    "# wandb.login()  # paste your free-tier API key when prompted, once per session"
)


# ============================================================ M3 ==========
def build_m3():
    cells = [
        md("# MENTORA — Train M3: Performance Trajectory Predictor (LSTM)\n\n"
           "Do this one first — no HuggingFace dependency, trains from scratch on "
           "the synthetic score sequences from Phase 3. Target: **val MAE <= 5**."),
        *SHARED_SETUP,
        pip_cell(),
        md("## 1. Load and window the data"),
        code(
            "import pandas as pd, numpy as np, torch\n"
            "from torch.utils.data import Dataset, DataLoader\n\n"
            "df = pd.read_csv(f'{DATA}/processed/m3/score_histories_no_label.csv')\n\n"
            "SEQ_LEN = 5   # feed 5 past sessions, predict the next score\n"
            "class TrajectoryDataset(Dataset):\n"
            "    def __init__(self, df, seq_len=SEQ_LEN):\n"
            "        self.samples = []\n"
            "        for student_id, group in df.groupby('student_id'):\n"
            "            scores = group.sort_values('session_number')['score'].values\n"
            "            for i in range(len(scores) - seq_len):\n"
            "                x = scores[i:i+seq_len]\n"
            "                y = scores[i+seq_len]\n"
            "                self.samples.append((x, y))\n"
            "    def __len__(self): return len(self.samples)\n"
            "    def __getitem__(self, idx):\n"
            "        x, y = self.samples[idx]\n"
            "        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)\n\n"
            "full_ds = TrajectoryDataset(df)\n"
            "n_val = int(0.15 * len(full_ds))\n"
            "train_ds, val_ds = torch.utils.data.random_split(full_ds, [len(full_ds) - n_val, n_val])\n"
            "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)\n"
            "val_loader = DataLoader(val_ds, batch_size=32)\n"
            "print(f'{len(train_ds)} train windows, {len(val_ds)} val windows')"
        ),
        md("## 2. Model definition"),
        code(
            "import torch.nn as nn\n\n"
            "class TrajectoryLSTM(nn.Module):\n"
            "    def __init__(self, input_size=1, hidden_size=128, num_layers=2):\n"
            "        super().__init__()\n"
            "        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n"
            "        self.fc = nn.Linear(hidden_size, 1)\n"
            "    def forward(self, x):\n"
            "        out, _ = self.lstm(x)\n"
            "        return self.fc(out[:, -1, :]).squeeze(-1)\n\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "model = TrajectoryLSTM().to(device)\n"
            "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
            "criterion = nn.MSELoss()"
        ),
        md("## 3. Training loop — checkpoints to Drive every epoch, resumable"),
        code(
            "CKPT_PATH = f'{MODELS}/model3_trajectory_predictor/checkpoint.pt'\n"
            "start_epoch = 0\n\n"
            "if os.path.exists(CKPT_PATH):\n"
            "    ckpt = torch.load(CKPT_PATH, map_location=device)\n"
            "    model.load_state_dict(ckpt['model_state'])\n"
            "    optimizer.load_state_dict(ckpt['optimizer_state'])\n"
            "    start_epoch = ckpt['epoch'] + 1\n"
            "    print(f'Resumed from epoch {start_epoch}')\n\n"
            "N_EPOCHS = 30\n"
            "for epoch in range(start_epoch, N_EPOCHS):\n"
            "    model.train()\n"
            "    train_loss = 0\n"
            "    for x, y in train_loader:\n"
            "        x, y = x.to(device), y.to(device)\n"
            "        optimizer.zero_grad()\n"
            "        pred = model(x)\n"
            "        loss = criterion(pred, y)\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        train_loss += loss.item() * x.size(0)\n"
            "    train_loss /= len(train_ds)\n\n"
            "    model.eval()\n"
            "    val_abs_errors = []\n"
            "    with torch.no_grad():\n"
            "        for x, y in val_loader:\n"
            "            x, y = x.to(device), y.to(device)\n"
            "            pred = model(x)\n"
            "            val_abs_errors.extend((pred - y).abs().cpu().tolist())\n"
            "    val_mae = np.mean(val_abs_errors)\n\n"
            "    print(f'epoch {epoch+1}/{N_EPOCHS} - train_loss {train_loss:.3f} - val_MAE {val_mae:.3f}')\n\n"
            "    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch},\n"
            "               CKPT_PATH)\n\n"
            "    if val_mae <= 5.0:\n"
            "        print(f'Target MAE <= 5 reached at epoch {epoch+1} (val_MAE={val_mae:.2f}) - can stop here')\n"
            "        break"
        ),
        md("## 4. Save final model"),
        code(
            "torch.save(model.state_dict(), f'{MODELS}/model3_trajectory_predictor/lstm_final.pt')\n"
            "print('M3 done. Target: MAE <= 5. Achieved:', val_mae)"
        ),
        md("## Target metric: MAE <= 5\n\nIf validation MAE plateaus above that:\n"
           "- Check archetype balance from Phase 3's validation checklist\n"
           "- Try `hidden_size=256` or a 3rd LSTM layer\n"
           "- Increase `SEQ_LEN` to 7-8 if most students have enough sessions"),
    ]
    return make_notebook(cells)


# ============================================================ M4 ==========
def build_m4():
    cells = [
        md("# MENTORA — Train M4: Career-Skill Semantic Matcher (Sentence-Transformers)\n\n"
           "Target: **Precision@5 >= 0.75**.\n\n"
           "**Schema note:** our `training_pairs.csv` has columns "
           "`(required_skills, job_title, label)` where `required_skills` is a "
           "stringified Python list, not the doc's `(skill_text, career_text, label)` — "
           "the load cell below adapts for this."),
        *SHARED_SETUP,
        pip_cell(),
        WANDB_CELL_MD, WANDB_CELL_CODE,
        md("## 1. Load base model + training pairs"),
        code(
            "from sentence_transformers import SentenceTransformer, InputExample, losses\n"
            "from torch.utils.data import DataLoader\n"
            "import pandas as pd, ast\n\n"
            "model = SentenceTransformer('all-mpnet-base-v2')  # downloads once, ~420MB\n"
            "pairs = pd.read_csv(f'{DATA}/processed/m4/training_pairs.csv')\n"
            "pairs['required_skills'] = pairs['required_skills'].apply(ast.literal_eval)\n"
            "pairs['skill_text'] = pairs['required_skills'].apply(lambda skills: ', '.join(skills))\n"
            "pairs = pairs.rename(columns={'job_title': 'career_text'})\n\n"
            "# held-out split at the ROW level here is fine since each row is one\n"
            "# (career, shuffled-or-own title) pair, not a repeated student -- no\n"
            "# near-duplicate leak risk like Phase 4 §8 warns about for other models\n"
            "train_pairs = pairs.sample(frac=0.85, random_state=42)\n"
            "val_pairs = pairs.drop(train_pairs.index)\n\n"
            "train_examples = [\n"
            "    InputExample(texts=[row['skill_text'], row['career_text']], label=float(row['label']))\n"
            "    for _, row in train_pairs.iterrows()\n"
            "]\n"
            "train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)\n"
            "train_loss = losses.CosineSimilarityLoss(model)"
        ),
        md("## 2. Precision@5 evaluator — the real target metric"),
        code(
            "from sentence_transformers.evaluation import SentenceEvaluator\n"
            "import numpy as np\n\n"
            "class PrecisionAtKEvaluator(SentenceEvaluator):\n"
            "    def __init__(self, skill_texts, career_texts, relevant_career_indices, k=5):\n"
            "        self.skill_texts = skill_texts\n"
            "        self.career_texts = career_texts\n"
            "        self.relevant = relevant_career_indices\n"
            "        self.k = k\n\n"
            "    def __call__(self, model, output_path=None, epoch=-1, steps=-1):\n"
            "        skill_emb = model.encode(self.skill_texts, convert_to_numpy=True)\n"
            "        career_emb = model.encode(self.career_texts, convert_to_numpy=True)\n"
            "        precisions = []\n"
            "        for i, s_emb in enumerate(skill_emb):\n"
            "            sims = career_emb @ s_emb / (np.linalg.norm(career_emb, axis=1) * np.linalg.norm(s_emb) + 1e-9)\n"
            "            top_k = np.argsort(-sims)[:self.k]\n"
            "            hits = len(set(top_k) & self.relevant[i])\n"
            "            precisions.append(hits / self.k)\n"
            "        score = float(np.mean(precisions))\n"
            "        print(f'[epoch {epoch}] Precision@{self.k} = {score:.3f}')\n"
            "        return score\n\n"
            "career_profiles = pd.read_csv(f'{DATA}/processed/m4/career_profiles.csv')\n"
            "career_profiles['required_skills'] = career_profiles['required_skills'].apply(ast.literal_eval)\n"
            "all_career_texts = career_profiles['title'].tolist()\n"
            "title_to_idx = {t: i for i, t in enumerate(all_career_texts)}\n\n"
            "val_positive = val_pairs[val_pairs['label'] == 1.0]\n"
            "val_skill_texts = val_positive['skill_text'].tolist()\n"
            "val_relevant_indices = [{title_to_idx[t]} for t in val_positive['career_text']]"
        ),
        md("## 3. Fine-tune"),
        code(
            "evaluator = PrecisionAtKEvaluator(val_skill_texts, all_career_texts, val_relevant_indices, k=5)\n\n"
            "model.fit(\n"
            "    train_objectives=[(train_dataloader, train_loss)],\n"
            "    epochs=4,\n"
            "    evaluator=evaluator,\n"
            "    evaluation_steps=200,\n"
            "    output_path=f'{MODELS}/model4_career_matcher',\n"
            "    save_best_model=True,\n"
            ")"
        ),
        md("## Target metric: Precision@5 >= 0.75\n\nIf it plateaus lower:\n"
           "- Add hard negatives (similar-sounding but non-matching career pairs) to `training_pairs.csv`\n"
           "- Try `epochs=8`\n"
           "- Consider swapping the starter `career_profiles.csv` for a real O*NET join (see datasets/SOURCES.md) — broader, more realistic skill vocabulary tends to help this metric directly"),
    ]
    return make_notebook(cells)


# ============================================================ M1 ==========
def build_m1():
    cells = [
        md("# MENTORA — Train M1: Knowledge Gap Classifier (DeBERTa-v3-base)\n\n"
           "Target: **F1 (micro) >= 0.85**.\n\n"
           "**Note:** our starter `question_bank.csv` has 70 questions total "
           "(target is 150-300 per subject — see datasets/SOURCES.md). Expect "
           "this run to be a pipeline smoke test more than a metric-hitting run "
           "until the bank is scaled up with real past-paper questions."),
        *SHARED_SETUP,
        pip_cell(),
        WANDB_CELL_MD, WANDB_CELL_CODE,
        md("## 1. Load and tokenize"),
        code(
            "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer\n"
            "from datasets import Dataset\n"
            "import pandas as pd, numpy as np\n\n"
            "df = pd.read_csv(f'{DATA}/processed/m1/question_bank.csv')\n"
            "topics = sorted(df['topic'].unique())\n"
            "NUM_TOPICS = len(topics)\n"
            "topic_to_idx = {t: i for i, t in enumerate(topics)}\n\n"
            "def to_multihot(topic_str):\n"
            "    vec = [0.0] * NUM_TOPICS\n"
            "    for t in str(topic_str).split('|'):  # our data is single-topic; this also handles multi-topic if added later\n"
            "        vec[topic_to_idx[t]] = 1.0\n"
            "    return vec\n\n"
            "df['labels'] = df['topic'].apply(to_multihot)\n\n"
            "tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')\n"
            "def tokenize(batch):\n"
            "    enc = tokenizer(batch['question_text'], truncation=True, padding='max_length', max_length=128)\n"
            "    enc['labels'] = batch['labels']\n"
            "    return enc\n\n"
            "ds = Dataset.from_pandas(df[['question_text', 'labels']])\n"
            "ds = ds.train_test_split(test_size=0.15, seed=42)\n"
            "ds = ds.map(tokenize, batched=True)"
        ),
        md("## 2. Model + metrics"),
        code(
            "model = AutoModelForSequenceClassification.from_pretrained(\n"
            "    'microsoft/deberta-v3-base', num_labels=NUM_TOPICS, problem_type='multi_label_classification'\n"
            ")\n\n"
            "from sklearn.metrics import f1_score\n\n"
            "def compute_metrics(eval_pred):\n"
            "    logits, labels = eval_pred\n"
            "    probs = 1 / (1 + np.exp(-logits))\n"
            "    preds = (probs >= 0.5).astype(int)\n"
            "    return {'f1_micro': f1_score(labels, preds, average='micro', zero_division=0)}"
        ),
        md("## 3. Train — checkpoints every epoch to Drive, resumable\n\n"
           "**Note:** `fp16` is off below. DeBERTa-v3's disentangled attention "
           "layer has a known incompatibility with PyTorch's `GradScaler` that "
           "throws `ValueError: Attempting to unscale FP16 gradients` partway "
           "through the first step — this isn't environment-specific, it's a "
           "model architecture issue. DeBERTa-base is small enough that fp32 "
           "training is still fine on a T4."),
        code(
            "import glob\n\n"
            "OUT_DIR = f'{MODELS}/model1_gap_classifier'\n\n"
            "args = TrainingArguments(\n"
            "    output_dir=OUT_DIR,\n"
            "    per_device_train_batch_size=8,\n"
            "    per_device_eval_batch_size=8,\n"
            "    num_train_epochs=5,\n"
            "    learning_rate=2e-5,\n"
            "    save_strategy='epoch',\n"
            "    eval_strategy='epoch',\n"
            "    load_best_model_at_end=True,\n"
            "    metric_for_best_model='f1_micro',\n"
            "    fp16=False,  # see note above -- DeBERTa-v3 + fp16 GradScaler incompatibility\n"
            "    report_to='wandb' if 'wandb' in dir() else 'none',\n"
            ")\n\n"
            "trainer = Trainer(\n"
            "    model=model, args=args,\n"
            "    train_dataset=ds['train'], eval_dataset=ds['test'],\n"
            "    compute_metrics=compute_metrics,\n"
            ")\n\n"
            "existing_checkpoints = glob.glob(f'{OUT_DIR}/checkpoint-*')\n"
            "resume = sorted(existing_checkpoints)[-1] if existing_checkpoints else None\n"
            "trainer.train(resume_from_checkpoint=resume)"
        ),
        md("## 4. Save final model"),
        code(
            "trainer.save_model(f'{OUT_DIR}/final')\n"
            "tokenizer.save_pretrained(f'{OUT_DIR}/final')\n"
            "print(trainer.evaluate())"
        ),
        md("## Target metric: F1 (micro) >= 0.85\n\nIf it plateaus lower:\n"
           "- Sweep the classification threshold (0.3/0.4/0.5/0.6) on validation\n"
           "- Check per-topic example counts — merge near-unlearnable rare subtopics into their parent\n"
           "- Highest-leverage fix: grow the M1 question bank past the 70-question starter set (also directly benefits M2)"),
    ]
    return make_notebook(cells)


# ============================================================ M5 ==========
def build_m5():
    cells = [
        md("# MENTORA — Train M5: CV/Transcript NER Extractor (BERT-base-uncased)\n\n"
           "Target: **Entity F1 >= 0.82** on the gold test split.\n\n"
           "**Schema note:** Phase 3 only produced `labeled/m5/gold_val_test.jsonl` "
           "(20 hand-authored, unambiguous resumes used directly as gold — no separate "
           "silver/auto-tagged training set yet, see datasets/SOURCES.md). This "
           "notebook splits gold 60/20/20 into train/val/test instead of the doc's "
           "silver-train/gold-val-test split. **TODO once real Kaggle resumes are "
           "added:** regenerate with a real silver set and switch back to "
           "train-on-silver, evaluate-on-gold, per the original Phase 3/4 design — "
           "that's the setup that gives a trustworthy F1 that isn't just the "
           "auto-tagger agreeing with itself."),
        *SHARED_SETUP,
        pip_cell(),
        md("## 1. Load and split the gold set (see schema note above)"),
        code(
            "import json, random\n"
            "from datasets import Dataset\n\n"
            "def load_jsonl(path):\n"
            "    rows = []\n"
            "    with open(path) as f:\n"
            "        for line in f:\n"
            "            rows.append(json.loads(line))\n"
            "    return rows\n\n"
            "gold = load_jsonl(f'{DATA}/labeled/m5/gold_val_test.jsonl')\n\n"
            "all_tags = sorted({tag for row in gold for tag in row['ner_tags']})\n"
            "tag2id = {t: i for i, t in enumerate(all_tags)}\n"
            "id2tag = {i: t for t, i in tag2id.items()}\n\n"
            "random.seed(42)\n"
            "random.shuffle(gold)\n"
            "n = len(gold)\n"
            "train, val, test = gold[:int(n*0.6)], gold[int(n*0.6):int(n*0.8)], gold[int(n*0.8):]\n"
            "print(f'{len(train)} train / {len(val)} val / {len(test)} test resumes')\n\n"
            "train_ds = Dataset.from_list(train)\n"
            "val_ds = Dataset.from_list(val)\n"
            "test_ds = Dataset.from_list(test)"
        ),
        md("## 2. Tokenize with label alignment"),
        code(
            "from transformers import AutoTokenizer\n\n"
            "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n\n"
            "def tokenize_and_align(batch):\n"
            "    tokenized = tokenizer(batch['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=256)\n"
            "    all_labels = []\n"
            "    for i, tags in enumerate(batch['ner_tags']):\n"
            "        word_ids = tokenized.word_ids(batch_index=i)\n"
            "        label_ids, prev_word = [], None\n"
            "        for word_id in word_ids:\n"
            "            if word_id is None:\n"
            "                label_ids.append(-100)\n"
            "            elif word_id != prev_word:\n"
            "                label_ids.append(tag2id[tags[word_id]])\n"
            "            else:\n"
            "                label_ids.append(-100)\n"
            "            prev_word = word_id\n"
            "        all_labels.append(label_ids)\n"
            "    tokenized['labels'] = all_labels\n"
            "    return tokenized\n\n"
            "train_ds = train_ds.map(tokenize_and_align, batched=True)\n"
            "val_ds = val_ds.map(tokenize_and_align, batched=True)\n"
            "test_ds = test_ds.map(tokenize_and_align, batched=True)"
        ),
        md("## 3. Model, metrics (entity F1 via seqeval), training"),
        code(
            "from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer\n"
            "import evaluate, numpy as np, glob\n\n"
            "model = AutoModelForTokenClassification.from_pretrained(\n"
            "    'bert-base-uncased', num_labels=len(all_tags), id2label=id2tag, label2id=tag2id\n"
            ")\n"
            "seqeval = evaluate.load('seqeval')\n\n"
            "def compute_metrics(eval_pred):\n"
            "    logits, labels = eval_pred\n"
            "    preds = np.argmax(logits, axis=2)\n"
            "    true_preds, true_labels = [], []\n"
            "    for pred_row, label_row in zip(preds, labels):\n"
            "        p_seq, l_seq = [], []\n"
            "        for p, l in zip(pred_row, label_row):\n"
            "            if l != -100:\n"
            "                p_seq.append(id2tag[p]); l_seq.append(id2tag[l])\n"
            "        true_preds.append(p_seq); true_labels.append(l_seq)\n"
            "    results = seqeval.compute(predictions=true_preds, references=true_labels)\n"
            "    return {'entity_f1': results['overall_f1']}\n\n"
            "OUT_DIR = f'{MODELS}/model5_cv_ner'\n"
            "args = TrainingArguments(\n"
            "    output_dir=OUT_DIR, per_device_train_batch_size=8, num_train_epochs=4,\n"
            "    save_strategy='epoch', eval_strategy='epoch', load_best_model_at_end=True,\n"
            "    metric_for_best_model='entity_f1', fp16=True,\n"
            ")\n"
            "trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)\n\n"
            "existing = sorted(glob.glob(f'{OUT_DIR}/checkpoint-*'))\n"
            "trainer.train(resume_from_checkpoint=existing[-1] if existing else None)"
        ),
        md("## 4. Save + evaluate on held-out gold test"),
        code(
            "trainer.save_model(f'{OUT_DIR}/final')\n"
            "tokenizer.save_pretrained(f'{OUT_DIR}/final')\n"
            "print('Test set (gold, held out):', trainer.evaluate(test_ds))"
        ),
        md("## Target metric: Entity F1 >= 0.82\n\nIf it plateaus lower:\n"
           "- Check per-entity-type F1 (`seqeval.compute(..., mode='strict')`) — usually one type (often CERT, smallest vocab) drags the average down\n"
           "- Scale up the resume corpus and, once real Kaggle resumes are added, restore the silver/gold split described in the schema note above"),
    ]
    return make_notebook(cells)


# ============================================================ M2 ==========
def build_m2():
    cells = [
        md("# MENTORA — Train M2: Adaptive Question Generator (FLAN-T5-Large + LoRA)\n\n"
           "Do this one last — heaviest lift. Target: **BLEU-4 >= 0.35, ROUGE-L >= 0.50**.\n\n"
           "**Note:** our `flan_t5_training_data.csv` currently has 70 prompt/target "
           "pairs (target is 500+ — see datasets/SOURCES.md; it scales automatically "
           "as M1's question bank grows). Expect this run to be a pipeline smoke test "
           "until then."),
        *SHARED_SETUP,
        pip_cell(),
        WANDB_CELL_MD, WANDB_CELL_CODE,
        md("## 1. Load base model + apply LoRA"),
        code(
            "from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer\n"
            "from peft import LoraConfig, get_peft_model, TaskType\n"
            "import pandas as pd, numpy as np, glob\n"
            "from datasets import Dataset\n\n"
            "tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')\n"
            "base_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')\n\n"
            "lora_config = LoraConfig(\n"
            "    task_type=TaskType.SEQ_2_SEQ_LM,\n"
            "    r=8, lora_alpha=16, lora_dropout=0.05,\n"
            "    target_modules=['q', 'v'],\n"
            ")\n"
            "model = get_peft_model(base_model, lora_config)\n"
            "model.print_trainable_parameters()"
        ),
        md("## 2. Load and tokenize the Phase 3 data"),
        code(
            "df = pd.read_csv(f'{DATA}/processed/m2/flan_t5_training_data.csv')\n"
            "ds = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)\n\n"
            "def tokenize(batch):\n"
            "    inputs = tokenizer(batch['input_text'], truncation=True, padding='max_length', max_length=96)\n"
            "    targets = tokenizer(batch['target_text'], truncation=True, padding='max_length', max_length=256)\n"
            "    inputs['labels'] = targets['input_ids']\n"
            "    return inputs\n\n"
            "ds = ds.map(tokenize, batched=True)"
        ),
        md("## 3. Metrics (BLEU-4 + ROUGE-L)"),
        code(
            "import evaluate\n"
            "sacrebleu = evaluate.load('sacrebleu')\n"
            "rouge = evaluate.load('rouge')\n\n"
            "def compute_metrics(eval_pred):\n"
            "    preds, labels = eval_pred\n"
            "    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)\n"
            "    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)\n"
            "    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)\n"
            "    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)\n"
            "    bleu = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])\n"
            "    rougeL = rouge.compute(predictions=decoded_preds, references=decoded_labels)['rougeL']\n"
            "    return {'bleu4': bleu['score'], 'rougeL': rougeL}"
        ),
        md("## 4. Training arguments — memory-conscious for a free T4"),
        code(
            "OUT_DIR = f'{MODELS}/model2_question_generator'\n"
            "args = Seq2SeqTrainingArguments(\n"
            "    output_dir=OUT_DIR,\n"
            "    per_device_train_batch_size=4,\n"
            "    gradient_accumulation_steps=4,\n"
            "    num_train_epochs=3,\n"
            "    learning_rate=1e-4,\n"
            "    predict_with_generate=True,\n"
            "    fp16=True,\n"
            "    save_strategy='steps',\n"
            "    save_steps=200,\n"
            "    eval_strategy='steps',\n"
            "    eval_steps=200,\n"
            "    load_best_model_at_end=True,\n"
            "    metric_for_best_model='rougeL',\n"
            "    report_to='wandb' if 'wandb' in dir() else 'none',\n"
            ")\n\n"
            "trainer = Seq2SeqTrainer(\n"
            "    model=model, args=args,\n"
            "    train_dataset=ds['train'], eval_dataset=ds['test'],\n"
            "    compute_metrics=compute_metrics,\n"
            ")\n\n"
            "existing = sorted(glob.glob(f'{OUT_DIR}/checkpoint-*'), key=lambda p: int(p.split('-')[-1]))\n"
            "trainer.train(resume_from_checkpoint=existing[-1] if existing else None)"
        ),
        md("## 5. Save — LoRA adapter only"),
        code(
            "model.save_pretrained(f'{OUT_DIR}/lora_adapter')\n"
            "tokenizer.save_pretrained(f'{OUT_DIR}/lora_adapter')\n"
            "print(trainer.evaluate())"
        ),
        md("## Target metric: BLEU-4 >= 0.35, ROUGE-L >= 0.50\n\nIf it plateaus lower:\n"
           "- Check `target_text` JSON is consistently formatted (Phase 3's script already validates this on generation)\n"
           "- Bump `r=16` in the LoRA config\n"
           "- Highest-leverage fix: grow M1's question bank (M2 reuses it directly)"),
    ]
    return make_notebook(cells)


for name, builder in [
    ("train_m3.ipynb", build_m3),
    ("train_m4.ipynb", build_m4),
    ("train_m1.ipynb", build_m1),
    ("train_m5.ipynb", build_m5),
    ("train_m2.ipynb", build_m2),
]:
    nb = builder()
    path = OUT_DIR / name
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")
