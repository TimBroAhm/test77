# =============================================================================
# MSc Thesis: Intelligent Code Review Outcome Prediction
# Full implementation – Colab ready
# Includes: preprocessing, 4 models, evaluation, cross‑validation,
#           Monte Carlo dropout, bias analysis, explainability (SHAP/LIME),
#           hyperparameter tuning (Optuna, optional), wandb logging.
# =============================================================================

# -------------------------------
# 1. Install dependencies (Colab only)
# -------------------------------
!pip install -q torch transformers datasets scikit-learn imbalanced-learn nltk spacy matplotlib seaborn pandas numpy optuna shap lime wandb tqdm scipy
!python -m spacy download en_core_web_sm

# -------------------------------
# 2. Imports and setup
# -------------------------------
import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup,
    RobertaTokenizer, RobertaModel
)

# Scikit‑learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
from scipy.stats import chi2_contingency

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameter tuning
import optuna
from optuna.trial import TrialState

# Explainability
import shap
from lime.lime_text import LimeTextExplainer

# Experiment tracking (optional)
import wandb

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------
# 3. Data Preprocessing Class
# -------------------------------
class DataPreprocessor:
    """Handles all data cleaning, tokenization, encoding and scaling.
       Fits encoders/scalers only on training data to avoid leakage."""

    def __init__(self, df):
        self.df = df.copy()
        self.label_encoder = LabelEncoder()
        self.lang_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_code_len = 512
        self.max_text_len = 128

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        text = text.lower().strip()
        return text

    def combine_text_fields(self):
        review_parts = []
        if 'all_comments' in self.df.columns:
            review_parts.append(self.df['all_comments'].fillna('').astype(str))
        if 'title' in self.df.columns:
            review_parts.append(self.df['title'].fillna('').astype(str))
        if 'body' in self.df.columns:
            review_parts.append(self.df['body'].fillna('').astype(str))
        self.df['review_text'] = review_parts[0]
        for part in review_parts[1:]:
            self.df['review_text'] += ' ' + part
        self.df['review_text'] = self.df['review_text'].apply(self.clean_text)

    def clean_code_diff(self):
        self.df['code_diff'] = self.df['code_diff'].fillna('').astype(str).apply(self.clean_text)

    def encode_labels(self):
        self.df['label'] = self.df['outcome'].astype(int)

    def fit_encode_language(self, train_idx):
        self.lang_encoder.fit(self.df.loc[train_idx, 'language'].fillna('unknown'))
        self.df['lang'] = self.lang_encoder.transform(self.df['language'].fillna('unknown'))

    def fit_scale_metadata(self, train_idx):
        meta_cols = ['additions', 'deletions', 'num_files_changed', 'review_duration']
        for col in meta_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df.loc[train_idx, col].median())
        self.scaler.fit(self.df.loc[train_idx, meta_cols])
        self.df[meta_cols] = self.scaler.transform(self.df[meta_cols])

    def tokenize_for_dl(self, indices=None):
        if indices is None:
            indices = self.df.index
        code_tokens = self.tokenizer(
            self.df.loc[indices, 'code_diff'].tolist(),
            padding=True,
            truncation=True,
            max_length=self.max_code_len,
            return_tensors='pt'
        )
        text_tokens = self.tokenizer(
            self.df.loc[indices, 'review_text'].tolist(),
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        return code_tokens, text_tokens

    def run(self, train_idx, test_idx):
        self.combine_text_fields()
        self.clean_code_diff()
        self.encode_labels()
        self.fit_encode_language(train_idx)
        self.fit_scale_metadata(train_idx)
        code_tokens_all, text_tokens_all = self.tokenize_for_dl()
        return {
            'df': self.df,
            'code_tokens': code_tokens_all,
            'text_tokens': text_tokens_all,
            'lang_encoder': self.lang_encoder,
            'scaler': self.scaler
        }

# -------------------------------
# 4. Load dataset and split
# -------------------------------
# ADJUST THIS PATH: either upload file or mount Drive
DATASET_PATH = '/content/advanced_full_synthetic_code_review_dataset.csv'
# If file is uploaded directly to Colab, use: 'advanced_full_synthetic_code_review_dataset.csv'

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Stratified split: 70% train, 15% val, 15% test
train_val_df, test_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['outcome']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.15/0.7, random_state=42, stratify=train_val_df['outcome']
)

preprocessor = DataPreprocessor(df)
processed = preprocessor.run(train_idx=train_df.index, test_idx=test_df.index)
df = processed['df']
code_tokens = processed['code_tokens']
text_tokens = processed['text_tokens']

train_idx = train_df.index
val_idx = val_df.index
test_idx = test_df.index

print("Label distribution:\n", df['label'].value_counts())

# -------------------------------
# 5. Prepare PyTorch tensors
# -------------------------------
meta_cols = ['additions', 'deletions', 'num_files_changed', 'review_duration']

# Training
train_code = code_tokens['input_ids'][train_idx]
train_code_mask = code_tokens['attention_mask'][train_idx]
train_text = text_tokens['input_ids'][train_idx]
train_text_mask = text_tokens['attention_mask'][train_idx]
train_meta = torch.tensor(df.loc[train_idx, meta_cols].values, dtype=torch.float32)
train_lang = torch.tensor(df.loc[train_idx, 'lang'].values, dtype=torch.long)
train_labels = torch.tensor(df.loc[train_idx, 'label'].values, dtype=torch.long)

# Validation
val_code = code_tokens['input_ids'][val_idx]
val_code_mask = code_tokens['attention_mask'][val_idx]
val_text = text_tokens['input_ids'][val_idx]
val_text_mask = text_tokens['attention_mask'][val_idx]
val_meta = torch.tensor(df.loc[val_idx, meta_cols].values, dtype=torch.float32)
val_lang = torch.tensor(df.loc[val_idx, 'lang'].values, dtype=torch.long)
val_labels = torch.tensor(df.loc[val_idx, 'label'].values, dtype=torch.long)

# Test
test_code = code_tokens['input_ids'][test_idx]
test_code_mask = code_tokens['attention_mask'][test_idx]
test_text = text_tokens['input_ids'][test_idx]
test_text_mask = text_tokens['attention_mask'][test_idx]
test_meta = torch.tensor(df.loc[test_idx, meta_cols].values, dtype=torch.float32)
test_lang = torch.tensor(df.loc[test_idx, 'lang'].values, dtype=torch.long)
test_labels = torch.tensor(df.loc[test_idx, 'label'].values, dtype=torch.long)

# Class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels.numpy()), y=train_labels.numpy())
loss_weight = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# -------------------------------
# 6. Dataset classes
# -------------------------------
class ReviewDataset(Dataset):
    def __init__(self, code_ids, code_mask, text_ids, text_mask, lang, meta, labels):
        self.code_ids = code_ids
        self.code_mask = code_mask
        self.text_ids = text_ids
        self.text_mask = text_mask
        self.lang = lang
        self.meta = meta
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'code_ids': self.code_ids[idx],
            'code_mask': self.code_mask[idx],
            'text_ids': self.text_ids[idx],
            'text_mask': self.text_mask[idx],
            'lang': self.lang[idx],
            'meta': self.meta[idx],
            'label': self.labels[idx]
        }

class CodeOnlyDataset(Dataset):
    def __init__(self, code_ids, code_mask, labels):
        self.code_ids = code_ids
        self.code_mask = code_mask
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {'code_ids': self.code_ids[idx], 'code_mask': self.code_mask[idx], 'label': self.labels[idx]}

class TextOnlyDataset(Dataset):
    def __init__(self, text_ids, labels):
        self.text_ids = text_ids
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {'text_ids': self.text_ids[idx], 'label': self.labels[idx]}

train_dataset = ReviewDataset(train_code, train_code_mask, train_text, train_text_mask,
                              train_lang, train_meta, train_labels)
val_dataset = ReviewDataset(val_code, val_code_mask, val_text, val_text_mask,
                            val_lang, val_meta, val_labels)
test_dataset = ReviewDataset(test_code, test_code_mask, test_text, test_text_mask,
                             test_lang, test_meta, test_labels)

train_code_only = CodeOnlyDataset(train_code, train_code_mask, train_labels)
val_code_only = CodeOnlyDataset(val_code, val_code_mask, val_labels)
test_code_only = CodeOnlyDataset(test_code, test_code_mask, test_labels)

train_text_only = TextOnlyDataset(train_text, train_labels)
val_text_only = TextOnlyDataset(val_text, val_labels)
test_text_only = TextOnlyDataset(test_text, test_labels)

# -------------------------------
# 7. Model Definitions
# -------------------------------
# 7.1 BiLSTM with Attention
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_layers=2,
                 num_classes=3, dropout=0.3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = (lstm_out * attn_weights).sum(dim=1)
        out = self.dropout(attended)
        logits = self.fc(out)
        return logits

# 7.2 CodeBERT (code only)
class CodeBERTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits

# 7.3 Hybrid Model (CodeBERT + RoBERTa + metadata + language)
class HybridModel(nn.Module):
    def __init__(self, num_languages, meta_dim, num_classes=3):
        super().__init__()
        self.code_bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.text_bert = RobertaModel.from_pretrained("roberta-base")
        self.lang_embed = nn.Embedding(num_languages, 16)
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        fusion_size = 768 + 768 + 16 + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, code_ids, code_mask, text_ids, text_mask, lang, meta):
        code_out = self.code_bert(input_ids=code_ids, attention_mask=code_mask)
        code_cls = code_out.last_hidden_state[:, 0, :]

        text_out = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        text_cls = text_out.last_hidden_state[:, 0, :]

        lang_emb = self.lang_embed(lang)
        meta_emb = self.meta_fc(meta)

        combined = torch.cat([code_cls, text_cls, lang_emb, meta_emb], dim=-1)
        logits = self.classifier(combined)
        return logits

# -------------------------------
# 8. Classical Baselines (TF‑IDF + SMOTE)
# -------------------------------
def prepare_classical_features(train_texts, test_texts, train_metrics, test_metrics):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    train_metrics_sparse = csr_matrix(train_metrics)
    test_metrics_sparse = csr_matrix(test_metrics)
    train_features = hstack([train_tfidf, train_metrics_sparse])
    test_features = hstack([test_tfidf, test_metrics_sparse])
    return train_features, test_features, vectorizer

train_texts_raw = df.loc[train_idx, 'review_text'].tolist()
test_texts_raw = df.loc[test_idx, 'review_text'].tolist()
train_metrics = df.loc[train_idx, meta_cols].values
test_metrics = df.loc[test_idx, meta_cols].values

train_features, test_features, tfidf_vec = prepare_classical_features(
    train_texts_raw, test_texts_raw, train_metrics, test_metrics
)

smote = SMOTE(random_state=42)
train_features_res, train_labels_res = smote.fit_resample(train_features, train_labels.numpy())

# -------------------------------
# 9. Training utilities
# -------------------------------
def get_dataloaders(batch_size, model_type='hybrid'):
    if model_type == 'hybrid':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif model_type == 'codebert':
        train_loader = DataLoader(train_code_only, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_code_only, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_code_only, batch_size=batch_size, shuffle=False)
    elif model_type == 'bilstm':
        train_loader = DataLoader(train_text_only, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_text_only, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_text_only, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Unknown model type")
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay,
                model_name='model', loss_weight=loss_weight, patience=3):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            if isinstance(model, HybridModel):
                code_ids = batch['code_ids'].to(device)
                code_mask = batch['code_mask'].to(device)
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                lang = batch['lang'].to(device)
                meta = batch['meta'].to(device)
                labels = batch['label'].to(device)
                logits = model(code_ids, code_mask, text_ids, text_mask, lang, meta)
            elif isinstance(model, CodeBERTClassifier):
                code_ids = batch['code_ids'].to(device)
                code_mask = batch['code_mask'].to(device)
                labels = batch['label'].to(device)
                logits = model(code_ids, code_mask)
            elif isinstance(model, BiLSTMAttention):
                text_ids = batch['text_ids'].to(device)
                labels = batch['label'].to(device)
                logits = model(text_ids)
            else:
                raise TypeError("Unknown model type")

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(model, HybridModel):
                    code_ids = batch['code_ids'].to(device)
                    code_mask = batch['code_mask'].to(device)
                    text_ids = batch['text_ids'].to(device)
                    text_mask = batch['text_mask'].to(device)
                    lang = batch['lang'].to(device)
                    meta = batch['meta'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(code_ids, code_mask, text_ids, text_mask, lang, meta)
                elif isinstance(model, CodeBERTClassifier):
                    code_ids = batch['code_ids'].to(device)
                    code_mask = batch['code_mask'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(code_ids, code_mask)
                elif isinstance(model, BiLSTMAttention):
                    text_ids = batch['text_ids'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(text_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_{model_name}.pt")
            print(f"  -> Saved best {model_name}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    model.load_state_dict(torch.load(f"best_{model_name}.pt"))
    return model

# -------------------------------
# 10. Evaluation function
# -------------------------------
def evaluate_model(model, test_loader, model_type='hybrid', class_names=['reject', 'merge', 'revise']):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'hybrid':
                code_ids = batch['code_ids'].to(device)
                code_mask = batch['code_mask'].to(device)
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                lang = batch['lang'].to(device)
                meta = batch['meta'].to(device)
                logits = model(code_ids, code_mask, text_ids, text_mask, lang, meta)
            elif model_type == 'codebert':
                code_ids = batch['code_ids'].to(device)
                code_mask = batch['code_mask'].to(device)
                logits = model(code_ids, code_mask)
            elif model_type == 'bilstm':
                text_ids = batch['text_ids'].to(device)
                logits = model(text_ids)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    per_class = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1,2])
    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Precision': per_class[0],
        'Recall': per_class[1],
        'F1-Score': per_class[2],
        'Support': per_class[3]
    })

    print(f"\n=== {model_type.upper()} Test Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"ROC-AUC (OvR): {auc:.4f}")
    print("\nPer-class metrics:")
    print(per_class_df.to_string(index=False))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'auc': auc,
        'per_class': per_class_df,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs
    }

# -------------------------------
# 11. (Optional) Hyperparameter tuning with Optuna
# -------------------------------
# Uncomment and run if you want to tune the hybrid model (may take time)
"""
def objective_hybrid(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)

    num_languages = len(processed['lang_encoder'].classes_)
    meta_dim = len(meta_cols)
    model = HybridModel(num_languages, meta_dim, num_classes=3).to(device)

    train_loader, val_loader, _ = get_dataloaders(batch_size, model_type='hybrid')
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    for epoch in range(5):
        model.train()
        for batch in train_loader:
            code_ids = batch['code_ids'].to(device)
            code_mask = batch['code_mask'].to(device)
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            lang = batch['lang'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['label'].to(device)
            logits = model(code_ids, code_mask, text_ids, text_mask, lang, meta)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                code_ids = batch['code_ids'].to(device)
                code_mask = batch['code_mask'].to(device)
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                lang = batch['lang'].to(device)
                meta = batch['meta'].to(device)
                labels = batch['label'].to(device)
                logits = model(code_ids, code_mask, text_ids, text_mask, lang, meta)
                val_loss += criterion(logits, labels).item()
        val_loss /= len(val_loader)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective_hybrid, n_trials=20)
print("Best hyperparameters for Hybrid:", study.best_params)
"""

# -------------------------------
# 12. Cross‑validation for Hybrid Model
# -------------------------------
def cross_validate_hybrid(df, code_tokens, text_tokens, num_folds=5, epochs=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n--- Fold {fold+1} ---")
        # Prepare data for this fold
        train_code_fold = code_tokens['input_ids'][train_idx]
        val_code_fold = code_tokens['input_ids'][val_idx]
        train_code_mask_fold = code_tokens['attention_mask'][train_idx]
        val_code_mask_fold = code_tokens['attention_mask'][val_idx]
        train_text_fold = text_tokens['input_ids'][train_idx]
        val_text_fold = text_tokens['input_ids'][val_idx]
        train_text_mask_fold = text_tokens['attention_mask'][train_idx]
        val_text_mask_fold = text_tokens['attention_mask'][val_idx]

        train_meta_fold = torch.tensor(df.loc[train_idx, meta_cols].values, dtype=torch.float32)
        val_meta_fold = torch.tensor(df.loc[val_idx, meta_cols].values, dtype=torch.float32)
        train_lang_fold = torch.tensor(df.loc[train_idx, 'lang'].values, dtype=torch.long)
        val_lang_fold = torch.tensor(df.loc[val_idx, 'lang'].values, dtype=torch.long)
        train_labels_fold = torch.tensor(df.loc[train_idx, 'label'].values, dtype=torch.long)
        val_labels_fold = torch.tensor(df.loc[val_idx, 'label'].values, dtype=torch.long)

        train_dataset_fold = ReviewDataset(
            train_code_fold, train_code_mask_fold,
            train_text_fold, train_text_mask_fold,
            train_lang_fold, train_meta_fold, train_labels_fold
        )
        val_dataset_fold = ReviewDataset(
            val_code_fold, val_code_mask_fold,
            val_text_fold, val_text_mask_fold,
            val_lang_fold, val_meta_fold, val_labels_fold
        )
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=16, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=16, shuffle=False)

        num_languages = len(processed['lang_encoder'].classes_)
        meta_dim = len(meta_cols)
        model_fold = HybridModel(num_languages, meta_dim, num_classes=3).to(device)
        model_fold = train_model(model_fold, train_loader_fold, val_loader_fold,
                                 epochs=epochs, lr=2e-5, weight_decay=0.01,
                                 model_name=f'hybrid_fold{fold}')

        res = evaluate_model(model_fold, val_loader_fold, model_type='hybrid')
        for k in metrics.keys():
            metrics[k].append(res[k])

    print("\n=== Cross‑validation results (mean ± std) ===")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    return metrics

# -------------------------------
# 13. Monte Carlo Dropout Uncertainty
# -------------------------------
def mc_dropout_predictions(model, test_loader, model_type='hybrid', n_iterations=50):
    model.train()
    all_probs = []
    with torch.no_grad():
        for _ in range(n_iterations):
            probs_iter = []
            for batch in test_loader:
                if model_type == 'hybrid':
                    code_ids = batch['code_ids'].to(device)
                    code_mask = batch['code_mask'].to(device)
                    text_ids = batch['text_ids'].to(device)
                    text_mask = batch['text_mask'].to(device)
                    lang = batch['lang'].to(device)
                    meta = batch['meta'].to(device)
                    logits = model(code_ids, code_mask, text_ids, text_mask, lang, meta)
                elif model_type == 'codebert':
                    code_ids = batch['code_ids'].to(device)
                    code_mask = batch['code_mask'].to(device)
                    logits = model(code_ids, code_mask)
                elif model_type == 'bilstm':
                    text_ids = batch['text_ids'].to(device)
                    logits = model(text_ids)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                probs_iter.append(probs)
            all_probs.append(np.concatenate(probs_iter, axis=0))
    all_probs = np.stack(all_probs)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    return mean_probs, std_probs

# -------------------------------
# 14. Bias Analysis
# -------------------------------
def bias_analysis(df, protected_attr='author', label_col='label'):
    author_counts = df[protected_attr].value_counts()
    rare_authors = author_counts[author_counts < 5].index
    df_clean = df.copy()
    df_clean.loc[df_clean[protected_attr].isin(rare_authors), protected_attr] = 'other'

    contingency = pd.crosstab(df_clean[protected_attr], df_clean[label_col])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\n=== Bias Analysis ===")
    print(f"Chi‑square test for {protected_attr}: p = {p:.4f}")
    if p < 0.05:
        print("  Significant difference found – potential bias.")
    else:
        print("  No significant difference detected.")
    return p

# -------------------------------
# 15. Explainability (SHAP & LIME)
# -------------------------------
def shap_global_explanation(model, test_loader, device, num_samples=100):
    """SHAP for CodeBERT (simplest)."""
    model.eval()
    batch = next(iter(test_loader))
    input_ids = batch['code_ids'][:num_samples].to(device)
    attention_mask = batch['code_mask'][:num_samples].to(device)

    def predict(inputs):
        logits = model(inputs[0], inputs[1])
        return F.softmax(logits, dim=-1).cpu().numpy()

    explainer = shap.GradientExplainer(model, [input_ids, attention_mask])
    shap_values = explainer.shap_values([input_ids, attention_mask])
    shap.summary_plot(shap_values, feature_names=None, show=False)
    plt.savefig('shap_summary_codebert.png')
    plt.show()

def lime_explain_instance(text, model, tokenizer, class_names=['reject', 'merge', 'revise'], model_type='bilstm'):
    def predict_proba(texts):
        model.eval()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            if model_type == 'bilstm':
                logits = model(inputs['input_ids'])
            else:
                logits = model(inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    exp.show_in_notebook()
    exp.save_to_file('lime_explanation.html')

# -------------------------------
# 16. Main execution
# -------------------------------
if __name__ == "__main__":
    # Initialize wandb (optional – comment out if not used)
    wandb.init(project='code-review-outcome', config={
        'model': 'Hybrid',
        'batch_size': 16,
        'lr': 2e-5,
        'epochs': 10,
        'weight_decay': 0.01,
        'class_weights': class_weights.tolist()
    })

    # 16.1 Classical Baselines
    print("\n=== Classical Baseline Results ===")
    classical_models = {
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for name, model in classical_models.items():
        model.fit(train_features_res, train_labels_res)
        preds = model.predict(test_features)
        acc = accuracy_score(test_labels.numpy(), preds)
        print(f"{name} Test Accuracy: {acc:.4f}")

    # 16.2 BiLSTM
    print("\n=== Training BiLSTM with Attention ===")
    vocab_size = preprocessor.tokenizer.vocab_size
    bilstm_model = BiLSTMAttention(vocab_size, hidden_dim=256, num_layers=2,
                                    num_classes=3, dropout=0.2).to(device)
    train_loader_bl, val_loader_bl, test_loader_bl = get_dataloaders(batch_size=32, model_type='bilstm')
    bilstm_model = train_model(bilstm_model, train_loader_bl, val_loader_bl,
                               epochs=10, lr=1e-3, weight_decay=0.01, model_name='bilstm')
    eval_bilstm = evaluate_model(bilstm_model, test_loader_bl, model_type='bilstm')

    # 16.3 CodeBERT
    print("\n=== Training CodeBERT (code only) ===")
    codebert_model = CodeBERTClassifier(num_classes=3).to(device)
    train_loader_cb, val_loader_cb, test_loader_cb = get_dataloaders(batch_size=16, model_type='codebert')
    codebert_model = train_model(codebert_model, train_loader_cb, val_loader_cb,
                                 epochs=10, lr=2e-5, weight_decay=0.01, model_name='codebert')
    eval_codebert = evaluate_model(codebert_model, test_loader_cb, model_type='codebert')

    # 16.4 Hybrid
    print("\n=== Training Hybrid Model ===")
    num_languages = len(processed['lang_encoder'].classes_)
    meta_dim = len(meta_cols)
    hybrid_model = HybridModel(num_languages, meta_dim, num_classes=3).to(device)
    train_loader_hy, val_loader_hy, test_loader_hy = get_dataloaders(batch_size=16, model_type='hybrid')
    hybrid_model = train_model(hybrid_model, train_loader_hy, val_loader_hy,
                               epochs=10, lr=2e-5, weight_decay=0.01, model_name='hybrid')
    eval_hybrid = evaluate_model(hybrid_model, test_loader_hy, model_type='hybrid')

    # 16.5 Cross‑validation (optional – uncomment to run)
    # cross_validate_hybrid(df, code_tokens, text_tokens, num_folds=5, epochs=5)

    # 16.6 Monte Carlo Dropout
    print("\n=== Monte Carlo Dropout Uncertainty (Hybrid Model) ===")
    mean_probs, std_probs = mc_dropout_predictions(hybrid_model, test_loader_hy, model_type='hybrid', n_iterations=50)
    print(f"Average prediction uncertainty (std): {std_probs.mean():.4f}")

    # 16.7 Bias Analysis
    bias_analysis(df, protected_attr='author', label_col='label')

    # 16.8 Explainability (optional – may be slow)
    # shap_global_explanation(codebert_model, test_loader_cb, device)
    # sample_text = df.loc[test_idx[0], 'review_text']
    # lime_explain_instance(sample_text, bilstm_model, preprocessor.tokenizer, model_type='bilstm')

    # 16.9 Generate result tables and plots
    results_summary = pd.DataFrame({
        'Model': ['BiLSTM', 'CodeBERT', 'Multimodal Fusion'],
        'Precision': [eval_bilstm['precision_weighted'], eval_codebert['precision_weighted'], eval_hybrid['precision_weighted']],
        'Recall': [eval_bilstm['recall_weighted'], eval_codebert['recall_weighted'], eval_hybrid['recall_weighted']],
        'F1-Score': [eval_bilstm['f1_weighted'], eval_codebert['f1_weighted'], eval_hybrid['f1_weighted']],
        'ROC-AUC': [eval_bilstm['auc'], eval_codebert['auc'], eval_hybrid['auc']],
        'Test Acc': [eval_bilstm['accuracy'], eval_codebert['accuracy'], eval_hybrid['accuracy']]
    })
    print("\n=== Model Comparison Table (LaTeX) ===")
    print(results_summary.to_latex(index=False, float_format="%.3f"))

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model_name, cm in zip(axes, ['BiLSTM', 'CodeBERT', 'Hybrid'],
                                   [eval_bilstm['confusion_matrix'],
                                    eval_codebert['confusion_matrix'],
                                    eval_hybrid['confusion_matrix']]):
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['reject','merge','revise'],
                    yticklabels=['reject','merge','revise'])
        ax.set_title(model_name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

    wandb.finish()
    print("\nAll experiments completed.")
