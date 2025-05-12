import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import time
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from wordcloud import WordCloud

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchtext 
from torchtext.vocab import Vocab 
from torchtext.data.utils import get_tokenizer
from collections import Counter 

# Ensure NLTK resources are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

os.makedirs("plots", exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


DATASETS = [
    {
        "filename": "Hate Speech Detection Dataset.csv",
        "text_candidates": ["Comment", "text", "body"],
        "label_candidates": ["Hateful", "label", "hate"],
        "name": "Hate Speech",
        "special": None,
        "max_rows": 2500
    },
    {
        "filename": "NIH Curated Data.csv",
        "text_candidates": ["Content", "comment", "Comment", "text", "body"],
        "label_candidates": ["label", "Hateful", "hate"],
        "name": "NIH Curated",
        "special": None,
        "max_rows": 1000
    },
    {
        "filename": "Ethos Multi-Label Dataset.csv",
        "text_candidates": ["comment", "Comment", "text", "body"],
        "label_candidates": ["any_hate"],
        "name": "Ethos Multi-Label",
        "special": "ethos_binary",
        "max_rows": 2500
    }
]

# Helper functions
def find_best_col(df, candidates):
    df_cols_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    raise ValueError(f"Could not find a suitable column among: {candidates} in DataFrame columns: {list(df.columns)}")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(token) for token in text.split())

def preprocess(text):
    if not isinstance(text, str): return ""
    text = clean_text(text)
    stop_words = stopwords.words('english')
    tokens = [t for t in text.split() if t not in stop_words]
    return lemmatize_text(" ".join(tokens))

def plot_wordcloud(series, title, fname):
    text_corpus = " ".join(series.dropna().astype(str))
    if text_corpus.strip():
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text_corpus)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    else:
        print(f"Skipping word cloud for {title} due to empty corpus.")

def plot_label_distribution(series, title, fname):
    if series.empty:
        print(f"Skipping label distribution plot for {title} due to empty series.")
        return
    plt.figure(figsize=(6,4))
    series.value_counts().sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_text_length(series, title, fname):
    if series.dropna().astype(str).empty:
        print(f"Skipping text length plot for {title} due to no text data.")
        return
    plt.figure(figsize=(8,5))
    series.dropna().astype(str).apply(len).hist(bins=40)
    plt.title(title)
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def ethos_binary(df):
    label_cols = [col for col in df.columns if col.lower() != "comment" and df[col].dtype in (np.number, 'bool')]
    if not label_cols:
        raise ValueError("No numeric/boolean label columns found for Ethos dataset processing, aside from 'comment'.")
    df['any_hate'] = (df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float) > 0.5).any(axis=1).astype(int)
    return df

def balanced_sample(df, label_col, max_rows_total, shuffle=True, allow_upsampling=True):
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(columns=df.columns)

    class_counts = df[label_col].value_counts()
    n_classes = len(class_counts)

    if n_classes == 0:
        return pd.DataFrame(columns=df.columns)

    n_per_class = max(1, int(max_rows_total / n_classes))
    
    sampled_dfs = []
    for cls_label, current_class_size in class_counts.items():
        df_cls = df[df[label_col] == cls_label]
        
        if current_class_size == 0:
            continue

        if allow_upsampling:
            num_to_sample = n_per_class
            use_replacement_for_sample = (n_per_class > current_class_size)
        else:
            num_to_sample = min(n_per_class, current_class_size)
            use_replacement_for_sample = False

        sampled_dfs.append(df_cls.sample(n=num_to_sample, random_state=SEED, replace=use_replacement_for_sample))

    if not sampled_dfs:
        return pd.DataFrame(columns=df.columns)

    final_df = pd.concat(sampled_dfs)
    if shuffle:
        final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return final_df

# EDA Process Summary
def run_eda(df, text_col, label_col, dataset_name_sanitized, eda_summary):
    print(f"\n--- EDA for {dataset_name_sanitized} ---")
    eda_summary['shape_after_initial_sampling'] = tuple(df.shape)
    eda_summary['columns'] = list(df.columns)
    
    if df.empty or text_col not in df.columns:
        print("EDA: DataFrame is empty or text column missing. Skipping.")
        return

    print(f"Sample text (first 2): {df[text_col].dropna().iloc[:2].tolist()}")

    if label_col in df.columns and not df[label_col].empty:
        plot_label_distribution(df[label_col], f"{dataset_name_sanitized} Class Distribution", f"plots/{dataset_name_sanitized}_class_dist.png")
        eda_summary['label_counts'] = df[label_col].value_counts().to_dict()
        print("Label counts:", eda_summary['label_counts'])
    else:
        print(f"Label column '{label_col}' not found or empty for EDA plots.")

    plot_text_length(df[text_col], f"{dataset_name_sanitized} Text Length Distribution", f"plots/{dataset_name_sanitized}_text_len.png")
    plot_wordcloud(df[text_col], f"{dataset_name_sanitized} Word Cloud", f"plots/{dataset_name_sanitized}_wordcloud.png")

    lens = df[text_col].dropna().astype(str).apply(len)
    if not lens.empty:
        eda_summary['len_mean'] = lens.mean()
        eda_summary['len_median'] = lens.median()
        eda_summary['len_min'] = lens.min()
        eda_summary['len_max'] = lens.max()
        eda_summary['length_stats_str'] = f"Mean={lens.mean():.1f}, Median={int(lens.median())}, Min={lens.min()}, Max={lens.max()}"
        print("Text length stats:", eda_summary['length_stats_str'])
    
    clean_corpus_series = df[text_col].dropna().astype(str).apply(clean_text)
    if not clean_corpus_series.empty:
        all_words = [word for text in clean_corpus_series for word in text.split()]
        if all_words:
            word_counts_series = pd.Series(all_words).value_counts()
            top20words = word_counts_series.head(20).to_dict()
            eda_summary['top20words'] = top20words
            print(f"Top 20 words (after basic cleaning): {list(top20words.keys())}")


def run_classical_modeling(df, text_col, label_col, dataset_name_sanitized, max_rows_sampling_for_model, model_summary):
    print(f"\n--- Classical Modeling for {dataset_name_sanitized} ---")
    start_time = time.time()

    df_model_input = df.dropna(subset=[text_col, label_col]).copy()
    if df_model_input.empty:
        print("No data for classical modeling after NaNs drop.")
        model_summary["classic_time"] = time.time() - start_time
        return None, None

    df_model_input["processed_text"] = df_model_input[text_col].apply(preprocess)
    
    df_sampled_for_model = balanced_sample(df_model_input, label_col, max_rows_sampling_for_model, shuffle=True, allow_upsampling=False)
    
    if df_sampled_for_model.empty:
        print("No data after sampling for classical modeling.")
        model_summary["classic_time"] = time.time() - start_time
        return None, None

    X = df_sampled_for_model["processed_text"]
    y = df_sampled_for_model[label_col].astype(int)
    model_summary['nrows_classical_modeling'] = len(df_sampled_for_model)

    if len(X) < 4 or len(y.unique()) < 2 :
        print("Not enough data or classes for classical modeling after processing and sampling.")
        model_summary["classic_time"] = time.time() - start_time
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    vectorizer_configs = {
        'BoW': CountVectorizer(ngram_range=(1, 1), min_df=2, max_features=3000),
        'TFIDF': TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000)
    }
    classifier_configs = {
        "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=300, random_state=SEED),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(random_state=SEED, max_iter=1000, dual=True)
    }
    
    results_list = []
    for v_name, vectorizer_obj in vectorizer_configs.items():
        print(f"Vectorizing with {v_name}...")
        X_train_vec = vectorizer_obj.fit_transform(X_train)
        X_test_vec = vectorizer_obj.transform(X_test)
        
        for c_name, classifier_obj in classifier_configs.items():
            print(f"Training {c_name} with {v_name}...")
            model = classifier_obj
            best_params_info = "Defaults"
            if c_name == "Logistic Regression":
                param_grid = {"C": [0.1, 1, 10]}
                cv_folds = min(3, y_train.value_counts().min()) 
                if cv_folds < 2: cv_folds = 2
                
                grid_search = GridSearchCV(classifier_obj, param_grid, cv=cv_folds, scoring="f1_weighted", n_jobs=-1)
                try:
                    grid_search.fit(X_train_vec, y_train)
                    model = grid_search.best_estimator_
                    best_params_info = str(grid_search.best_params_)
                except ValueError as e:
                    print(f"GridSearchCV failed for {c_name} ({v_name}): {e}. Using default parameters.")
                    model.fit(X_train_vec, y_train)

            else:
                model.fit(X_train_vec, y_train)
            
            y_pred = model.predict(X_test_vec)
            
            y_probs = None
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test_vec)[:, 1]
            elif hasattr(model, "decision_function"):
                dec_func = model.decision_function(X_test_vec)
                if dec_func.ndim > 1 and dec_func.shape[1] > 1:
                    dec_func = dec_func[:, 1]
                y_probs = (dec_func - dec_func.min()) / (dec_func.max() - dec_func.min() + 1e-8)
            
            roc_auc_val = roc_auc_score(y_test, y_probs) if y_probs is not None and len(np.unique(y_test)) > 1 else None
            
            results_list.append({
                'Classifier': c_name, 'Vectorizer': v_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0, average='weighted'),
                'Recall': recall_score(y_test, y_pred, zero_division=0, average='weighted'),
                'F1': f1_score(y_test, y_pred, zero_division=0, average='weighted'),
                'ROC_AUC': roc_auc_val,
                'Best_Params': best_params_info
            })
            print(f"\n> {c_name} ({v_name}):")
            print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    results_df = pd.DataFrame(results_list)
    model_summary["classic_time"] = time.time() - start_time
    
    if not results_df.empty:
        best_model_row = results_df.loc[results_df['F1'].idxmax()]
        model_summary['best_classical_model_stats'] = best_model_row.to_dict()
        print(f"\n>> Best Classical Model for {dataset_name_sanitized}: {best_model_row['Classifier']} ({best_model_row['Vectorizer']}), F1={best_model_row['F1']:.3f}")
        print("\nFull Classical Results Table:")
        print(results_df[['Classifier','Vectorizer','Accuracy','Precision','Recall','F1','ROC_AUC','Best_Params']].to_string())
        model_summary['classic_results_table'] = results_df.to_dict('records')
    else:
        print("No classical model results generated.")
        model_summary['best_classical_model_stats'] = None
        
    print(f"Classical modeling time: {model_summary['classic_time']:.1f}s")
    return results_df, model_summary.get('best_classical_model_stats')

# --- PyTorch Components ---
def yield_tokens_for_vocab(texts_iterator, tokenizer_func):
    for text_item in texts_iterator:
        yield tokenizer_func(str(text_item))

# Builds a custom vocab from the texts iterable.
def build_custom_vocab(texts_iterable, tokenizer_func, min_freq=2):
    print(f"Building vocabulary with min_freq={min_freq}...")
    counter = Counter()
    token_count = 0
    processed_texts_count = 0
    for tokens in yield_tokens_for_vocab(texts_iterable, tokenizer_func):
        counter.update(tokens)
        token_count += len(tokens)
        processed_texts_count +=1
    
    print(f"Processed {processed_texts_count} texts containing {token_count} total tokens for vocab. Unique tokens before min_freq: {len(counter)}.")
    
    specials_list = ['<pad>', '<unk>'] 
    
    vocab = Vocab(counter, 
                  min_freq=min_freq, 
                  specials=specials_list, 
                  specials_first=True)
    
    print(f"Vocabulary built. Size: {len(vocab)}. stoi['<pad>']={vocab.stoi['<pad>']}, stoi['<unk>']={vocab.stoi['<unk>']}")
    return vocab


class TextClassificationDataset(Dataset):
    def __init__(self, texts_list, labels_list, vocabulary, tokenizer_func, max_seq_length):
        self.texts = texts_list
        self.labels = labels_list
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer_func
        self.max_seq_length = max_seq_length
        self.pad_idx = vocabulary.stoi['<pad>'] 
        self.unk_idx = vocabulary.stoi['<unk>']


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        tokens = self.tokenizer(text)
        token_ids = [self.vocabulary.stoi.get(token, self.unk_idx) for token in tokens]

        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            padding_needed = self.max_seq_length - len(token_ids)
            token_ids.extend([self.pad_idx] * padding_needed)
            
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# define and test the LSTM model 
class LSTMHateSpeechClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout_rate, pad_idx_val, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=pad_idx_val)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx_val)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout_rate if n_layers > 1 else 0,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, text_batch):
        embedded = self.dropout(self.embedding(text_batch))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden).squeeze(1)

def run_lstm_on_dataset(df, text_col, label_col, dataset_name_sanitized, model_summary, 
                        max_rows_dl=2200, seq_len_val=50, emb_dim_val=100, batch_size_val=64, 
                        num_epochs=3):
    print(f"\n--- PyTorch LSTM Modeling for {dataset_name_sanitized} ---")
    start_time = time.time()

    df_model_input = df.dropna(subset=[text_col, label_col]).copy()
    if df_model_input.empty:
        print("No data for LSTM modeling after NaNs drop.")
        model_summary["lstm_time"] = time.time() - start_time
        return None
        
    df_model_input["processed_text"] = df_model_input[text_col].apply(preprocess)
    
    df_sampled_for_dl = balanced_sample(df_model_input, label_col, max_rows_dl, shuffle=True, allow_upsampling=True)

    if df_sampled_for_dl.empty:
        print("No data after sampling for LSTM modeling.")
        model_summary["lstm_time"] = time.time() - start_time
        return None

    X = df_sampled_for_dl["processed_text"].values
    y = df_sampled_for_dl[label_col].astype(int).values
    model_summary['nrows_lstm_modeling'] = len(df_sampled_for_dl)

    if len(X) < 2 or len(np.unique(y)) < 2:
        print("Not enough data or classes for LSTM modeling after processing and sampling for train/test split.")
        model_summary["lstm_time"] = time.time() - start_time
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Train or test split resulted in zero samples. Cannot proceed with LSTM.")
        model_summary["lstm_time"] = time.time() - start_time
        return None

    tokenizer_func = get_tokenizer("basic_english")
    vocab_obj = build_custom_vocab(X_train, tokenizer_func, min_freq=2)
    
    train_dataset = TextClassificationDataset(X_train, y_train, vocab_obj, tokenizer_func, seq_len_val)
    test_dataset = TextClassificationDataset(X_test, y_test, vocab_obj, tokenizer_func, seq_len_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

    INPUT_DIM = len(vocab_obj)
    EMBEDDING_DIM = emb_dim_val
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3
    PAD_IDX = vocab_obj.stoi['<pad>']

    initial_embeddings = torch.randn(INPUT_DIM, EMBEDDING_DIM)
    initial_embeddings[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    model = LSTMHateSpeechClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
                                     BIDIRECTIONAL, DROPOUT, PAD_IDX, pretrained_embeddings=initial_embeddings).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss().to(device)

    print(f"LSTM Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        if len(train_loader) == 0:
            print("Train loader is empty, skipping training epoch.")
            break
        for i, (batch_texts, batch_labels) in enumerate(train_loader):
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_texts)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (i+1) % max(1, len(train_loader)//5) == 0:
                 print(f'\rEpoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}', end='')
        if len(train_loader) > 0:
            print(f'\nEpoch {epoch+1} Average Train Loss: {epoch_loss/len(train_loader):.4f}')
        else:
            print(f'\nEpoch {epoch+1} had no training data.')


    model.eval()
    all_predictions_binary = []
    all_true_labels = []
    all_probabilities = []

    if len(test_loader) == 0:
        print("Test loader is empty. Skipping evaluation.")
        model_summary['lstm_time'] = time.time() - start_time
        model_summary['best_lstm_model_stats'] = {"Error": "Test loader empty"}
        return None

    with torch.no_grad():
        for batch_texts, batch_labels in test_loader:
            batch_texts = batch_texts.to(device)
            raw_predictions = model(batch_texts)
            probabilities = torch.sigmoid(raw_predictions).cpu().numpy()
            binary_predictions = (probabilities >= 0.5).astype(int)
            
            all_probabilities.extend(probabilities)
            all_predictions_binary.extend(binary_predictions)
            all_true_labels.extend(batch_labels.cpu().numpy().astype(int))

    if not all_true_labels:
        print("No data processed for LSTM evaluation.")
        model_summary['lstm_time'] = time.time() - start_time
        model_summary['best_lstm_model_stats'] = {"Error": "No evaluation data"}
        return None

    acc = accuracy_score(all_true_labels, all_predictions_binary)
    prec = precision_score(all_true_labels, all_predictions_binary, zero_division=0, average='weighted')
    rec = recall_score(all_true_labels, all_predictions_binary, zero_division=0, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions_binary, zero_division=0, average='weighted')
    roc_auc = roc_auc_score(all_true_labels, all_probabilities) if len(np.unique(all_true_labels)) > 1 else None
    
    print("\nLSTM Classification Report (Test Set):")
    print(classification_report(all_true_labels, all_predictions_binary, digits=3, zero_division=0))
    
    model_summary['lstm_time'] = time.time() - start_time
    lstm_results = {
        "Classifier": "LSTM (PyTorch)",
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc_auc,
        "nrows_trained_on": len(df_sampled_for_dl), 
        "params": {"seq_len": seq_len_val, "emb_dim": emb_dim_val, "epochs": num_epochs, 
                   "batch_size": batch_size_val, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
                   "bidirectional": BIDIRECTIONAL, "dropout": DROPOUT}
    }
    model_summary['best_lstm_model_stats'] = lstm_results
    print(f"LSTM Results: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, ROC_AUC={roc_auc if roc_auc else 'N/A':.3f}")
    print(f"LSTM modeling time: {model_summary['lstm_time']:.1f}s")
    return lstm_results

# --- Main Execution ---
print("="*42)
print("== NLP & Hate Speech Classification with PyTorch LSTM and Classic ML ==")
print("="*42)

overall_findings = []

for dataset_config in DATASETS:
    dataset_name = dataset_config['name']
    print(f"\n{'='*30}\nProcessing Dataset: {dataset_name}\n{'='*30}")
    
    dataset_start_time = time.time()
    current_summary = {"dataset_name": dataset_name, 
                       "max_rows_config_for_dataset": dataset_config.get('max_rows', 'Not specified')}
    
    try:
        df_raw_original = pd.read_csv(dataset_config['filename'], 
                                      sep=';' if "Ethos" in dataset_config['filename'] else ',',
                                      on_bad_lines='skip')
        current_summary['shape_raw_original'] = df_raw_original.shape
        print(f"Original raw data shape: {df_raw_original.shape}")

        df_for_processing = df_raw_original.copy()

        if 'max_rows' in dataset_config and dataset_config['name'] == "NIH Curated":
            sample_n = min(dataset_config['max_rows'], len(df_for_processing))
            if len(df_for_processing) > sample_n:
                print(f"Applying initial sampling to {dataset_name}: {sample_n} rows from {len(df_for_processing)}.")
                df_for_processing = df_for_processing.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
            current_summary['shape_after_initial_dataset_sampling'] = df_for_processing.shape
        else:
            current_summary['shape_after_initial_dataset_sampling'] = df_for_processing.shape

        text_col_name = find_best_col(df_for_processing, dataset_config['text_candidates'])
        
        if dataset_config.get('special') == "ethos_binary":
            df_processed_labels = ethos_binary(df_for_processing.copy())
            label_col_name = "any_hate"
        else:
            df_processed_labels = df_for_processing.copy()
            label_col_name = find_best_col(df_processed_labels, dataset_config['label_candidates'])

        current_summary['text_column_used'] = text_col_name
        current_summary['label_column_used'] = label_col_name

        run_eda(df_processed_labels, text_col_name, label_col_name, dataset_name.replace(" ", "_"), current_summary)
        
        max_rows_for_models = dataset_config.get('max_rows', 2500)

        _, _ = run_classical_modeling(df_processed_labels, text_col_name, label_col_name, 
                                                  dataset_name.replace(" ", "_"), 
                                                  max_rows_for_models, current_summary)
        
        _ = run_lstm_on_dataset(df_processed_labels, text_col_name, label_col_name, 
                                          dataset_name.replace(" ", "_"), current_summary,
                                          max_rows_dl=max_rows_for_models)
                                          
    except FileNotFoundError:
        print(f"ERROR: File not found: {dataset_config['filename']}")
        current_summary['error'] = f"File not found: {dataset_config['filename']}"
    except ValueError as ve:
        print(f"ERROR processing {dataset_name}: {ve}")
        current_summary['error'] = str(ve)
    except Exception as e:
        print(f"An unexpected error occurred with dataset {dataset_name}: {e}")
        current_summary['error'] = f"Unexpected error: {str(e)}"
        import traceback
        traceback.print_exc()

    current_summary['total_time_for_dataset'] = time.time() - dataset_start_time
    print(f"\nTime elapsed for {dataset_name}: {current_summary['total_time_for_dataset']:.1f}s")
    overall_findings.append(current_summary)

print("\n" + "="*40 + "\nFINAL PROJECT SUMMARY\n" + "="*40)
for s in overall_findings:
    print(f"\nDataset: {s['dataset_name']}")
    if 'error' in s:
        print(f"  Error during processing: {s['error']}")
        continue
    
    print(f"  Original Raw Shape: {s.get('shape_raw_original', 'N/A')}")
    print(f"  Shape after any initial sampling (used for EDA/Models): {s.get('shape_after_initial_dataset_sampling', 'N/A')}")
    print(f"  Max Rows Config for this dataset (for internal model sampling, or initial for NIH): {s.get('max_rows_config_for_dataset', 'N/A')}")
    print(f"  Text Col: {s.get('text_column_used', 'N/A')}, Label Col: {s.get('label_column_used', 'N/A')}")
    if 'length_stats_str' in s: print(f"  EDA Text Length Stats: {s['length_stats_str']}")
    if 'label_counts' in s: print(f"  EDA Label Counts: {s['label_counts']}")
    if 'top20words' in s: print(f"  EDA Top 20 Words: {list(s['top20words'].keys())}")
    
    if 'best_classical_model_stats' in s and s['best_classical_model_stats']:
        classic_stats = s['best_classical_model_stats']
        if "Error" not in classic_stats:
            print(f"  Best Classical Model ({classic_stats.get('Classifier')} w/ {classic_stats.get('Vectorizer')} on {s.get('nrows_classical_modeling','N/A')} rows):")
            print(f"    F1: {classic_stats.get('F1', 0):.3f}, Acc: {classic_stats.get('Accuracy', 0):.3f}, ROC_AUC: {classic_stats.get('ROC_AUC', 0) if classic_stats.get('ROC_AUC') is not None else 'N/A':.3f}")
            print(f"    Time: {s.get('classic_time', 0):.1f}s")
        else:
            print(f"  Classical Modeling Error: {classic_stats['Error']}")
    else:
        print("  Classical modeling results not available or error occurred.")
        
    if 'best_lstm_model_stats' in s and s['best_lstm_model_stats']:
        lstm_stats = s['best_lstm_model_stats']
        if "Error" not in lstm_stats:
            print(f"  LSTM Model (on {s.get('nrows_lstm_modeling','N/A')} rows):")
            print(f"    F1: {lstm_stats.get('F1', 0):.3f}, Acc: {lstm_stats.get('Accuracy', 0):.3f}, ROC_AUC: {lstm_stats.get('ROC_AUC', 0) if lstm_stats.get('ROC_AUC') is not None else 'N/A':.3f}")
            print(f"    Time: {s.get('lstm_time', 0):.1f}s")
        else:
             print(f"  LSTM Modeling Error: {lstm_stats['Error']}")
    else:
        print("  LSTM modeling results not available or error occurred.")
    print(f"  Total time for this dataset: {s.get('total_time_for_dataset', 0):.1f}s")

print("""
\nOverall Summary Notes:
- This script processes multiple datasets for hate speech classification.
- It performs Exploratory Data Analysis (EDA), classical machine learning, and LSTM-based deep learning.
- Key steps include text preprocessing, feature extraction (BoW, TF-IDF for classical; embeddings for LSTM), model training, and evaluation.
- Balanced sampling is used to handle class imbalance and manage dataset size for modeling stages.
- Specific datasets like NIH can be initially sampled to a smaller size for faster overall processing.
- All EDA plots are saved in the 'plots/' directory.
""")

print("All tasks finished! Check the 'plots/' directory for visualizations and the console output for detailed results.")

