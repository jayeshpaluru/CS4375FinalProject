import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)  # Required for lemmatization

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Enhanced function to preprocess text
def preprocess_text(text, remove_stopwords=True, stem=False, lemmatize=True):
    """
    Comprehensive text preprocessing function that performs:
    1. Lowercasing
    2. Punctuation removal
    3. Tokenization
    4. Stop-word removal (optional)
    5. Stemming (optional) or Lemmatization (optional)
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        stem (bool): Whether to apply stemming
        lemmatize (bool): Whether to apply lemmatization (ignored if stem=True)
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Get stopwords
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming or lemmatization
    if stem:
        tokens = [stemmer.stem(token) for token in tokens]
    elif lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return " ".join(tokens)

# Function to clean text (maintaining backward compatibility)
def clean_text(text):
    """Simple text cleaning for basic analysis"""
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

# Function to compare preprocessing methods
def compare_preprocessing_methods(text_series, output_dir="plots"):
    """Compare different preprocessing methods on sample texts"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Take a sample of texts
    sample_size = min(5, len(text_series))
    sample_texts = text_series.sample(sample_size)
    
    # Apply different preprocessing methods
    results = []
    for i, text in enumerate(sample_texts):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
            
        # Truncate text if too long
        display_text = text[:100] + "..." if len(text) > 100 else text
        
        results.append({
            "Original": display_text,
            "Cleaned": clean_text(text),
            "Tokenized": clean_text(text),
            "Stopwords Removed": preprocess_text(text, remove_stopwords=True, stem=False, lemmatize=False),
            "Stemmed": preprocess_text(text, remove_stopwords=True, stem=True, lemmatize=False),
            "Lemmatized": preprocess_text(text, remove_stopwords=True, stem=False, lemmatize=True)
        })
    
    # Save comparison to CSV
    pd.DataFrame(results).to_csv(f"{output_dir}/preprocessing_comparison.csv", index=False)
    
    print(f"Preprocessing comparison saved to {output_dir}/preprocessing_comparison.csv")
    
    # Return a brief sample for display
    return pd.DataFrame(results).head(2)

# Function to analyze a dataset
def analyze_dataset(df, text_column, label_column=None, dataset_name="Dataset", preprocess=True):
    print(f"\n{'='*50}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*50}")
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Clean and preprocess text
    stop_words = set(stopwords.words('english'))
    
    # Standard cleaning (for backward compatibility)
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Apply enhanced preprocessing
    if preprocess:
        print("\nApplying enhanced text preprocessing...")
        df['preprocessed_text'] = df[text_column].apply(lambda x: 
            preprocess_text(x, remove_stopwords=True, stem=False, lemmatize=True))
        
        # Show preprocessing comparison
        comparison_df = compare_preprocessing_methods(df[text_column])
        print("\nPreprocessing Comparison (Sample):")
        print(comparison_df)
    
    # Text length analysis (using preprocessed text if available)
    text_col_for_analysis = 'preprocessed_text' if preprocess and 'preprocessed_text' in df.columns else 'cleaned_text'
    
    df['word_count'] = df[text_col_for_analysis].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df['char_count'] = df[text_col_for_analysis].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Text length statistics
    print("\nText Length Statistics:")
    print(f"Words - Mean: {df['word_count'].mean():.2f}, Median: {df['word_count'].median()}, Max: {df['word_count'].max()}")
    print(f"Characters - Mean: {df['char_count'].mean():.2f}, Median: {df['char_count'].median()}, Max: {df['char_count'].max()}")
    
    # Plot text length distributions
    plot_text_length_distribution(df['word_count'], f'{dataset_name} - Word Count Distribution', f'{dataset_name.lower().replace(" ", "_")}_word_count_dist')
    plot_text_length_distribution(df['char_count'], f'{dataset_name} - Character Count Distribution', f'{dataset_name.lower().replace(" ", "_")}_char_count_dist')
    
    # Vocabulary analysis
    all_word_counts = get_word_counts(df[text_col_for_analysis], stop_words)
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
        # Rest of the code remains the same for class distribution analysis
        # ...
        pass

# Function to plot text length distributions (unchanged)
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

# Load Reddit dataset and apply preprocessing
print("Loading Reddit dataset...")
reddit_df = pd.read_csv("Reddit Sample Data.csv")

# Special focus on Reddit dataset preprocessing
print("\nApplying preprocessing to Reddit dataset...")
analyze_dataset(reddit_df, 'body', dataset_name="Reddit Dataset", preprocess=True)

# Save preprocessed data
reddit_df['preprocessed_text'] = reddit_df['body'].apply(lambda x: 
    preprocess_text(x, remove_stopwords=True, stem=False, lemmatize=True))
reddit_df.to_csv("Reddit_Sample_Data_Preprocessed.csv", index=False)

print("\nPreprocessing completed! Preprocessed Reddit data saved to 'Reddit_Sample_Data_Preprocessed.csv'")
print("Check the 'plots' folder for visualizations and preprocessing comparison.")