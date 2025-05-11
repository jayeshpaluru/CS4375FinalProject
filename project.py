import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
import os

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to get word counts
def get_word_counts(texts, stop_words=None):
    words = []
    for text in texts:
        if isinstance(text, str):
            words.extend([word for word in text.split() if word not in stop_words])
    return Counter(words)

# Function to generate wordcloud
def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate_from_frequencies(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', bbox_inches='tight')
    plt.close()

# Function to plot top N words
def plot_top_words(word_counts, title, filename, n=20):
    top_words = dict(word_counts.most_common(n))
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(top_words)), list(top_words.values()), align='center')
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.title(title, fontsize=16)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', bbox_inches='tight')
    plt.close()

# Function to plot text length distributions
def plot_text_length_distribution(lengths, title, filename, bins=50):
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=bins, alpha=0.7, color='steelblue')
    plt.title(title, fontsize=16)
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(lengths), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(lengths):.2f}')
    plt.axvline(np.median(lengths), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(lengths):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', bbox_inches='tight')
    plt.close()

# Function to analyze a dataset
def analyze_dataset(df, text_column, label_column=None, dataset_name="Dataset"):
    print(f"\n{'='*50}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*50}")
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Clean and preprocess text
    stop_words = set(stopwords.words('english'))
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Text length analysis
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df['char_count'] = df['cleaned_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Text length statistics
    print("\nText Length Statistics:")
    print(f"Words - Mean: {df['word_count'].mean():.2f}, Median: {df['word_count'].median()}, Max: {df['word_count'].max()}")
    print(f"Characters - Mean: {df['char_count'].mean():.2f}, Median: {df['char_count'].median()}, Max: {df['char_count'].max()}")
    
    # Plot text length distributions
    plot_text_length_distribution(df['word_count'], f'{dataset_name} - Word Count Distribution', f'{dataset_name.lower().replace(" ", "_")}_word_count_dist')
    plot_text_length_distribution(df['char_count'], f'{dataset_name} - Character Count Distribution', f'{dataset_name.lower().replace(" ", "_")}_char_count_dist')
    
    # Vocabulary analysis
    all_word_counts = get_word_counts(df['cleaned_text'], stop_words)
    unique_words = len(all_word_counts)
    total_words = sum(all_word_counts.values())
    
    print(f"\nVocabulary Analysis:")
    print(f"Total Words: {total_words}")
    print(f"Unique Words: {unique_words}")
    
    # Generate wordcloud and plot top words
    generate_wordcloud(all_word_counts, f'{dataset_name} - Word Cloud', f'{dataset_name.lower().replace(" ", "_")}_wordcloud')
    plot_top_words(all_word_counts, f'{dataset_name} - Top 20 Words', f'{dataset_name.lower().replace(" ", "_")}_top_words')
    
    # Class distribution (if label column is provided)
    if label_column and label_column in df.columns:
        print("\nClass Distribution:")
        class_dist = df[label_column].value_counts(normalize=True) * 100
        print(class_dist)
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        class_counts = df[label_column].value_counts()
        bars = plt.bar(class_counts.index.astype(str), class_counts.values)
        plt.title(f'{dataset_name} - Class Distribution', fontsize=16)
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.annotate(f'{height} ({class_dist.values[i]:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name.lower().replace(" ", "_")}_class_dist.png', bbox_inches='tight')
        plt.close()
        
        # Class-specific wordclouds
        class_values = df[label_column].unique()
        for class_val in class_values:
            class_texts = df[df[label_column] == class_val]['cleaned_text']
            class_word_counts = get_word_counts(class_texts, stop_words)
            
            if class_word_counts:  # Only generate if there are words
                generate_wordcloud(class_word_counts, 
                                  f'{dataset_name} - Class {class_val} Word Cloud', 
                                  f'{dataset_name.lower().replace(" ", "_")}_class_{class_val}_wordcloud')
                
                plot_top_words(class_word_counts, 
                              f'{dataset_name} - Class {class_val} Top 20 Words', 
                              f'{dataset_name.lower().replace(" ", "_")}_class_{class_val}_top_words')

# Load datasets
print("Loading datasets...")

# Hate Speech Detection Dataset
hate_speech_df = pd.read_csv("Hate Speech Detection Dataset.csv")
# Reddit Sample Data
reddit_df = pd.read_csv("Reddit Sample Data.csv") 
# Ethos Multi-Label Dataset
ethos_df = pd.read_csv("Ethos Multi-Label Dataset.csv", sep=';')

# Analyze each dataset
analyze_dataset(hate_speech_df, 'Comment', 'Hateful', "Hate Speech Dataset")

# For Reddit, we need to check if there are labels
if 'Hateful' in reddit_df.columns:
    analyze_dataset(reddit_df, 'body', 'Hateful', "Reddit Dataset")
else:
    analyze_dataset(reddit_df, 'body', dataset_name="Reddit Dataset")

# For Ethos, it's a multi-label dataset
ethos_columns = ethos_df.columns.tolist()
ethos_text_column = ethos_columns[0]  # First column should be text
ethos_label_columns = ethos_columns[1:]  # Rest are label columns

# Analyze Ethos dataset overall
analyze_dataset(ethos_df, ethos_text_column, dataset_name="Ethos Dataset")

# Convert multi-label to single binary label (any hate speech)
if len(ethos_label_columns) > 0:
    ethos_df['any_hate'] = ethos_df[ethos_label_columns].sum(axis=1) > 0
    analyze_dataset(ethos_df, ethos_text_column, 'any_hate', "Ethos Binary Dataset")
    
    # Analyze individual hate categories
    for label in ethos_label_columns:
        category_name = label.replace('_', ' ').title()
        ethos_subset = ethos_df[ethos_df[label] > 0]
        if len(ethos_subset) > 0:
            print(f"\nAnalyzing Ethos - {category_name} Category (n={len(ethos_subset)})")
            category_word_counts = get_word_counts(ethos_subset[ethos_text_column].apply(clean_text), 
                                                 set(stopwords.words('english')))
            
            generate_wordcloud(category_word_counts, 
                              f'Ethos - {category_name} Word Cloud', 
                              f'ethos_{label}_wordcloud')
            
            plot_top_words(category_word_counts, 
                          f'Ethos - {category_name} Top 20 Words', 
                          f'ethos_{label}_top_words')

print("\nEDA completed! Check the 'plots' folder for visualizations.")
