# MENTORA — Phase 4 Deep Dive: Training the Five Models on Google Colab

*Prerequisite: Phase 3 is done — all five datasets sit in
`mentora_data/processed/{m1..m5}/` and `mentora_data/labeled/m5/` on Google Drive,
validated. This phase produces one trained model per notebook, saved back to Drive,
ready for Phase 5's export/quantization step. Nothing in this phase touches the local
8GB laptop — everything runs on Colab's free T4 GPU. Estimated time: 2–4 weeks,
dominated by M2 (FLAN-T5-Large), with M3/M4 done in the first couple of days.*

---

## 0. One shared setup, reused across all five notebooks

Use **one notebook per model** (`train_m3.ipynb` … `train_m2.ipynb`), but every one of
them starts with the same first three cells. Write this once, copy it into each
notebook.

### 0.1 Cell 1 — mount Drive, confirm GPU
```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
# Expect to see "Tesla T4" — if this errors or shows no GPU, go to
# Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save, then re-run.
```

### 0.2 Cell 2 — folder scaffold (idempotent — safe to re-run after a disconnect)
```python
import os
DATA = '/content/drive/MyDrive/mentora_data'
MODELS = '/content/drive/MyDrive/mentora_models'
for m in ['model1_gap_classifier', 'model2_question_generator',
          'model3_trajectory_predictor', 'model4_career_matcher', 'model5_cv_ner']:
    os.makedirs(f'{MODELS}/{m}', exist_ok=True)
```

### 0.3 Cell 3 — install dependencies (unpinned first — see the master plan's §0.4)
```python
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121  # Colab has CUDA — use the GPU wheel here, NOT the CPU one from your laptop plan
!pip install -q transformers peft accelerate sentence-transformers datasets evaluate seqeval spacy sacrebleu rouge_score
```
Colab's GPU PyTorch build is unrelated to your laptop's CPU-only build from the master
plan — don't reuse that install command here. Training happens on Colab's CUDA GPU;
only *inference* later happens on your CPU-only laptop.

### 0.4 Optional — Weights & Biases logging
```python
!pip install -q wandb
import wandb
wandb.login()  # paste your free-tier API key when prompted, once per Colab session
```
Nice to have for watching loss curves live and for a screenshot in your thesis; skip
it entirely if you'd rather keep things simpler — nothing below strictly needs it.

### 0.5 The one habit that prevents lost work
**Every `Trainer`/training loop below saves a checkpoint to Drive, not to Colab's
local `/content/` disk, and every one is written to resume from that checkpoint if the
session disconnects mid-training.** Free Colab sessions disconnect without warning —
after ~12 hours, on idle timeout, or just because GPU demand is high that day. Treat
disconnection as something that *will* happen, not a rare edge case.

---

## 1. Training order (recap) and why

**M3 → M4 → M1 → M5 → M2.** Cheapest and fastest first, so you have *something*
trained and working end-to-end early, momentum stays up, and you can start Phase 5's
export tooling on a small model before tackling the fiddlier FLAN-T5 export later.

---

## 2. M3 — Performance Trajectory Predictor (LSTM) — do this first

### 2.1 Why this one has no HuggingFace dependency
It's a small, hand-written PyTorch model — 2-layer LSTM, 128 hidden units — trained
from scratch on the synthetic score sequences from Phase 3. No pretrained weights, no
tokenizer, no `Trainer` class. This is deliberately the simplest model in the whole
project.

### 2.2 Load and window the data
```python
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv(f'{DATA}/processed/m3/score_histories_no_label.csv')

SEQ_LEN = 5   # feed 5 past sessions, predict the next score
class TrajectoryDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.samples = []
        for student_id, group in df.groupby('student_id'):
            scores = group.sort_values('session_number')['score'].values
            for i in range(len(scores) - seq_len):
                x = scores[i:i+seq_len]
                y = scores[i+seq_len]
                self.samples.append((x, y))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

full_ds = TrajectoryDataset(df)
n_val = int(0.15 * len(full_ds))
train_ds, val_ds = torch.utils.data.random_split(full_ds, [len(full_ds) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
print(f"{len(train_ds)} train windows, {len(val_ds)} val windows")
```

### 2.3 Model definition
```python
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TrajectoryLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
```

### 2.4 Training loop with checkpoint-every-epoch and resume support
```python
import os

CKPT_PATH = f'{MODELS}/model3_trajectory_predictor/checkpoint.pt'
start_epoch = 0

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

N_EPOCHS = 30
for epoch in range(start_epoch, N_EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_abs_errors = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_abs_errors.extend((pred - y).abs().cpu().tolist())
    val_mae = np.mean(val_abs_errors)

    print(f"epoch {epoch+1}/{N_EPOCHS} — train_loss {train_loss:.3f} — val_MAE {val_mae:.3f}")

    # checkpoint EVERY epoch, straight to Drive
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch},
               CKPT_PATH)

    if val_mae <= 5.0:
        print(f"Target MAE ≤ 5 reached at epoch {epoch+1} (val_MAE={val_mae:.2f}) — can stop here")
        break
```
If the runtime disconnects at epoch 17, re-running Cells 1–3 then this cell picks up
from epoch 18 automatically — that's the whole point of the `if os.path.exists(...)`
resume block.

### 2.5 Save the final model
```python
torch.save(model.state_dict(), f'{MODELS}/model3_trajectory_predictor/lstm_final.pt')
print("M3 done. Target: MAE ≤ 5. Achieved:", val_mae)
```

### 2.6 Target metric and what to do if you miss it
**Target: MAE ≤ 5.** If validation MAE plateaus above that:
- Check the `archetype` balance from Phase 3's validation checklist — if one
  archetype dominates, the model will underfit the others.
- Try `hidden_size=256` or a 3rd LSTM layer — this model is small enough that Colab's
  free tier tolerates a much bigger one than you'd think.
- Increase `SEQ_LEN` to 7–8 if most students have enough sessions for it — more
  context per prediction usually helps trajectory models specifically.

---

## 3. M4 — Career-Skill Semantic Matcher (Sentence-Transformers)

### 3.1 Load the base model and the training pairs
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

model = SentenceTransformer('all-mpnet-base-v2')  # downloads once, ~420MB
pairs = pd.read_csv(f'{DATA}/processed/m4/training_pairs.csv')
# expected columns from Phase 3 §3: skill_text, career_text, label (1.0 = match, 0.0 = non-match)

train_examples = [
    InputExample(texts=[row['skill_text'], row['career_text']], label=float(row['label']))
    for _, row in pairs.iterrows()
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
```

### 3.2 Build a Precision@5 evaluator (the actual target metric, not just training loss)
`CosineSimilarityLoss` optimizes toward the *label* directly, but the metric you
actually need to hit is **Precision@5** — of the top-5 careers the fine-tuned model
ranks for a given skill set, how many are true matches. Write a small custom
evaluator:
```python
from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np

class PrecisionAtKEvaluator(SentenceEvaluator):
    def __init__(self, skill_texts, career_texts, relevant_career_indices, k=5):
        self.skill_texts = skill_texts          # one skill-profile string per validation student
        self.career_texts = career_texts         # the full career catalogue (all 50+)
        self.relevant = relevant_career_indices  # list of sets: ground-truth relevant career indices per student
        self.k = k

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        skill_emb = model.encode(self.skill_texts, convert_to_numpy=True)
        career_emb = model.encode(self.career_texts, convert_to_numpy=True)
        precisions = []
        for i, s_emb in enumerate(skill_emb):
            sims = career_emb @ s_emb / (np.linalg.norm(career_emb, axis=1) * np.linalg.norm(s_emb) + 1e-9)
            top_k = np.argsort(-sims)[:self.k]
            hits = len(set(top_k) & self.relevant[i])
            precisions.append(hits / self.k)
        score = float(np.mean(precisions))
        print(f"[epoch {epoch}] Precision@{self.k} = {score:.3f}")
        return score
```
Build `skill_texts`/`career_texts`/`relevant_career_indices` from a held-out slice of
`training_pairs.csv` (students you didn't train on, with their known-correct career
matches) plus `career_profiles.csv` for the full catalogue.

### 3.3 Fine-tune
```python
evaluator = PrecisionAtKEvaluator(val_skill_texts, all_career_texts, val_relevant_indices, k=5)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    evaluator=evaluator,
    evaluation_steps=200,
    output_path=f'{MODELS}/model4_career_matcher',
    save_best_model=True,   # keeps the checkpoint with the best Precision@5, not just the last epoch
)
```
`save_best_model=True` combined with `output_path` pointing straight at Drive is this
model's version of "checkpoint every epoch" — Sentence-Transformers handles this
internally, no manual resume logic needed here (short training, low disconnect risk;
if it does disconnect, just re-run — 4 epochs on this data is fast enough to redo).

### 3.4 Target metric and what to do if you miss it
**Target: Precision@5 ≥ 0.75.** If it plateaus lower:
- Check `training_pairs.csv` has enough *hard negatives* (careers that sound similar
  but aren't matches, e.g. "Data Scientist" vs. "Data Analyst" as a non-match pair) —
  a dataset of only obvious positives/negatives teaches the model too little.
- Try `epochs=8` — Sentence-Transformers fine-tunes tend to benefit from a bit more
  than the tech doc's original 4-epoch estimate on a first attempt.

---

## 4. M1 — Knowledge Gap Classifier (DeBERTa-v3-base)

### 4.1 Load and tokenize
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd, numpy as np

df = pd.read_csv(f'{DATA}/processed/m1/question_bank.csv')
topics = sorted(df['topic'].unique())
NUM_TOPICS = len(topics)
topic_to_idx = {t: i for i, t in enumerate(topics)}

def to_multihot(topic_list):
    vec = [0.0] * NUM_TOPICS
    for t in topic_list.split('|'):   # adjust the separator to match how Phase 3 stored multi-topic tags
        vec[topic_to_idx[t]] = 1.0
    return vec

df['labels'] = df['topic'].apply(to_multihot)

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
def tokenize(batch):
    enc = tokenizer(batch['question_text'], truncation=True, padding='max_length', max_length=128)
    enc['labels'] = batch['labels']
    return enc

ds = Dataset.from_pandas(df[['question_text', 'labels']])
ds = ds.train_test_split(test_size=0.15, seed=42)
ds = ds.map(tokenize, batched=True)
```

### 4.2 Model + metrics
```python
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base', num_labels=NUM_TOPICS, problem_type="multi_label_classification"
)

from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))       # sigmoid, since this is multi-label not softmax
    preds = (probs >= 0.5).astype(int)
    return {"f1_micro": f1_score(labels, preds, average="micro", zero_division=0)}
```

### 4.3 TrainingArguments with epoch checkpointing + resume
```python
OUT_DIR = f'{MODELS}/model1_gap_classifier'

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    fp16=True,               # Colab's T4 supports fp16 — meaningfully faster training
    report_to="wandb" if 'wandb' in dir() else "none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds['train'], eval_dataset=ds['test'],
    compute_metrics=compute_metrics,
)

import glob
existing_checkpoints = glob.glob(f'{OUT_DIR}/checkpoint-*')
resume = sorted(existing_checkpoints)[-1] if existing_checkpoints else None
trainer.train(resume_from_checkpoint=resume)
```
`save_strategy="epoch"` writes directly into `output_dir`, which is already pointed
at Drive — so `resume_from_checkpoint` picking up the latest `checkpoint-*` folder
works exactly the same way after a disconnect as it does the first time through.

### 4.4 Save the final model
```python
trainer.save_model(f'{OUT_DIR}/final')
tokenizer.save_pretrained(f'{OUT_DIR}/final')
print(trainer.evaluate())
```

### 4.5 Target metric and what to do if you miss it
**Target: F1 (micro) ≥ 0.85.** If it plateaus lower:
- Multi-label F1 is very sensitive to the 0.5 threshold — sweep thresholds
  (`0.3, 0.4, 0.5, 0.6`) on the validation set and report the best one; this alone
  often recovers several F1 points without retraining.
- Check topic-label imbalance — if some topics have very few examples, consider
  merging rare subtopics into their parent topic rather than leaving them
  near-unlearnable.

---

## 5. M5 — CV/Transcript NER Extractor (BERT-base-uncased)

### 5.1 Load gold (validation/test) and silver (train) sets
```python
import json
from datasets import Dataset

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

silver = load_jsonl(f'{DATA}/labeled/m5/silver_train.jsonl')   # training split
gold = load_jsonl(f'{DATA}/labeled/m5/gold_val_test.jsonl')     # validation + test split

# figure out the full tag set across both files
all_tags = sorted({tag for row in silver + gold for tag in row['ner_tags']})
tag2id = {t: i for i, t in enumerate(all_tags)}
id2tag = {i: t for t, i in tag2id.items()}

import random
random.seed(42)
random.shuffle(gold)
val, test = gold[:len(gold)//2], gold[len(gold)//2:]

train_ds = Dataset.from_list(silver)
val_ds = Dataset.from_list(val)
test_ds = Dataset.from_list(test)
```

### 5.2 Tokenize with label alignment (the one genuinely fiddly part of NER training)
BERT's wordpiece tokenizer splits some words into multiple sub-tokens — labels need
to align to those sub-tokens, not the original words:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_align(batch):
    tokenized = tokenizer(batch['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=256)
    all_labels = []
    for i, tags in enumerate(batch['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, prev_word = [], None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)          # special tokens ([CLS]/[SEP]/padding) — ignored in loss
            elif word_id != prev_word:
                label_ids.append(tag2id[tags[word_id]])
            else:
                label_ids.append(-100)           # only label the FIRST sub-token of each word
            prev_word = word_id
        all_labels.append(label_ids)
    tokenized['labels'] = all_labels
    return tokenized

train_ds = train_ds.map(tokenize_and_align, batched=True)
val_ds = val_ds.map(tokenize_and_align, batched=True)
test_ds = test_ds.map(tokenize_and_align, batched=True)
```

### 5.3 Model, metrics (entity-level F1 via `seqeval`), training
```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate

model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(all_tags), id2label=id2tag, label2id=tag2id
)
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)
    true_preds, true_labels = [], []
    for pred_row, label_row in zip(preds, labels):
        p_seq, l_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                p_seq.append(id2tag[p]); l_seq.append(id2tag[l])
        true_preds.append(p_seq); true_labels.append(l_seq)
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {"entity_f1": results["overall_f1"]}

OUT_DIR = f'{MODELS}/model5_cv_ner'
args = TrainingArguments(
    output_dir=OUT_DIR, per_device_train_batch_size=8, num_train_epochs=4,
    save_strategy="epoch", eval_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="entity_f1", fp16=True,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)

existing = sorted(glob.glob(f'{OUT_DIR}/checkpoint-*'))
trainer.train(resume_from_checkpoint=existing[-1] if existing else None)

print("Test set (gold, held out):", trainer.evaluate(test_ds))
trainer.save_model(f'{OUT_DIR}/final')
tokenizer.save_pretrained(f'{OUT_DIR}/final')
```
Note the split discipline from Phase 3: **train on silver (auto-tagged), evaluate on
gold (hand-corrected)** — this means your reported F1 reflects real label quality,
not the auto-tagger agreeing with itself.

### 5.4 Target metric and what to do if you miss it
**Target: Entity F1 ≥ 0.82.** If it plateaus lower:
- Check per-entity-type F1 (`seqeval.compute(..., mode="strict")` gives a breakdown)
  — usually one entity type (often `CERT`, since it has the smallest vocabulary in
  Phase 3's auto-tagger) drags the average down; expand that entity's gold examples
  specifically rather than the whole dataset.
- Silver-label noise is the other usual culprit — tighten the `PhraseMatcher`
  vocabulary from Phase 3 §5.3 and regenerate the silver set if spot-checks showed
  >30% error rate there.

---

## 6. M2 — Adaptive Question Generator (FLAN-T5-Large + LoRA) — do this last

### 6.1 Why this one is heavier, and the memory tactics that make it fit on a free T4
FLAN-T5-Large is ~770M parameters — too big to fine-tune fully on a free T4's 16GB
VRAM at reasonable batch sizes. LoRA fixes this by freezing the base model and
training small rank-decomposition adapter matrices (<1% of total parameters), which
is exactly why your Static Objectives allow this model specifically to stay
full-size while still being trainable here.

### 6.2 Load base model + apply LoRA
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
from datasets import Dataset

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
base_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q", "v"],
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # sanity check — expect well under 1% trainable
```

### 6.3 Load and tokenize the Phase 3 data
```python
df = pd.read_csv(f'{DATA}/processed/m2/flan_t5_training_data.csv')
ds = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)

def tokenize(batch):
    inputs = tokenizer(batch['input_text'], truncation=True, padding='max_length', max_length=96)
    targets = tokenizer(batch['target_text'], truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = targets['input_ids']
    return inputs

ds = ds.map(tokenize, batched=True)
```

### 6.4 Metrics (BLEU-4 + ROUGE-L)
```python
import evaluate
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    rougeL = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
    return {"bleu4": bleu["score"], "rougeL": rougeL}
```

### 6.5 Training arguments — memory-conscious for a free T4
```python
OUT_DIR = f'{MODELS}/model2_question_generator'
args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch size 16, without needing the VRAM for it
    num_train_epochs=3,
    learning_rate=1e-4,
    predict_with_generate=True,
    fp16=True,
    save_strategy="steps",
    save_steps=200,                       # checkpoint by STEPS not epochs — a 3-epoch FLAN-T5-Large
                                            # run on a free T4 can easily span multiple Colab sessions,
                                            # so don't wait a whole epoch between saves
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    report_to="wandb" if 'wandb' in dir() else "none",
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=ds['train'], eval_dataset=ds['test'],
    compute_metrics=compute_metrics,
)

existing = sorted(glob.glob(f'{OUT_DIR}/checkpoint-*'), key=lambda p: int(p.split('-')[-1]))
trainer.train(resume_from_checkpoint=existing[-1] if existing else None)
```

### 6.6 Save — LoRA adapter only, kept separate from the frozen base
```python
model.save_pretrained(f'{OUT_DIR}/lora_adapter')  # this is small (tens of MB), NOT the full 770M-param model
tokenizer.save_pretrained(f'{OUT_DIR}/lora_adapter')
print(trainer.evaluate())
```
Saving only the adapter (not `merge_and_unload()`-ing yet) is deliberate — Phase 5's
ONNX export step is where the adapter gets merged into the base weights, once, right
before export. Keeping them separate here means you can keep iterating on the LoRA
training without re-touching the (much larger, slower to move around) base model.

### 6.7 Target metric and what to do if you miss it
**Target: BLEU-4 ≥ 0.35, ROUGE-L ≥ 0.50.** If it plateaus lower:
- This is usually a data-shape problem before it's a training-hyperparameter
  problem — check that `target_text` (Phase 3 §6.2) is *consistently* formatted JSON;
  inconsistent formatting (sometimes quoted differently, sometimes missing a field)
  actively confuses a generation model.
- Bump `r=16` in the LoRA config for more adapter capacity if the trainable-param
  count from §6.2 feels too small relative to your dataset size.
- 500 examples (Phase 3's minimum) is genuinely on the small side for a generation
  task — if you have time, growing the M1 question bank (which M2 reuses, per Phase
  3 §6.1) is the highest-leverage fix available.

---

## 7. Experiment tracking — a simple log to keep across all five

Keep one shared spreadsheet or markdown table (in the repo, not just in your head) as
you go:

| Model | Attempt | Key hyperparams changed | Target metric | Achieved | Notes |
|---|---|---|---|---|---|
| M3 | 1 | hidden_size=128, 30 epochs | MAE ≤ 5 | — | — |
| M4 | 1 | 4 epochs, r=8 | P@5 ≥ 0.75 | — | — |
| M1 | 1 | lr=2e-5, 5 epochs | F1 ≥ 0.85 | — | — |
| M5 | 1 | 4 epochs | Entity F1 ≥ 0.82 | — | — |
| M2 | 1 | r=8, 3 epochs | BLEU-4 ≥ 0.35 / ROUGE-L ≥ 0.50 | — | — |

This table is also directly reusable in Phase 10's thesis chapter ("model training
results — targets vs. achieved").

---

## 8. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Colab disconnects, next run starts from epoch 0 again | Checkpoint was written to `/content/` instead of `/content/drive/MyDrive/...` | Double-check `output_dir`/`CKPT_PATH` in every training cell points into Drive, per §0.2 |
| `CUDA out of memory` on M2 | Batch size too high for FLAN-T5-Large on a T4 | Lower `per_device_train_batch_size` to 2, raise `gradient_accumulation_steps` to keep the effective batch size the same |
| Free Colab GPU quota exhausted mid-week | Ran too many long sessions back to back | Space out heavy training (especially M2) across days; M3/M4/M1/M5 combined use far less quota than M2 alone |
| Eval metric looks great on train, terrible on val | Train/val split leaked (e.g. the same question appears in both, reworded) | Re-check the Phase 3 split was done at the *student*/*source-document* level, not the row level, for datasets where near-duplicates are possible |
| `resume_from_checkpoint` errors with a shape mismatch | Changed `num_labels`/`LoraConfig` between runs without clearing old checkpoints | Delete the old `checkpoint-*` folders when you deliberately change model architecture, not just hyperparameters |
| `seqeval`/`sacrebleu` import errors | Forgot to `pip install evaluate seqeval sacrebleu rouge_score` in a fresh session | Re-run Cell 3 (§0.3) — Colab does NOT persist installed packages across sessions |

---

## 9. Definition of done

- [ ] M3: `lstm_final.pt` saved, val MAE ≤ 5 logged
- [ ] M4: best-checkpoint Sentence-Transformers folder saved, Precision@5 ≥ 0.75 logged
- [ ] M1: `final/` folder (model + tokenizer) saved, F1 (micro) ≥ 0.85 logged on held-out test
- [ ] M5: `final/` folder saved, Entity F1 ≥ 0.82 logged on the **gold** test split specifically
- [ ] M2: `lora_adapter/` folder saved (adapter only, not merged), BLEU-4 ≥ 0.35 and ROUGE-L ≥ 0.50 logged
- [ ] The §7 experiment table filled in for every model
- [ ] Every saved artifact lives under `mentora_models/` on Google Drive

Once every row above is checked, Phase 5 (export + ONNX quantization for local
inference) can start — download each `mentora_models/model*/` folder as the raw
material for that phase.
