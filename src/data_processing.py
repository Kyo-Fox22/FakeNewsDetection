import pandas as pd
import sentencepiece as spm
import os
from sklearn.model_selection import train_test_split

def combine_author(df: pd.DataFrame, author_col: str, content_col: str, 
                   out_col: str | None = None, separator: str = ': ') -> pd.DataFrame:
    """Combine an author column and a content column under one column using a separator.

    Args:
        df (pd.DataFrame): Origin of the dataset which author column and content column are located
        author_col (str): Name of the author column in the dataset.
        content_col (str): Name of the content column in the dataset.
        out_col (str | None, optional): Name of the new column to be created. If None, uses content_col as the new column. Defaults to None.
        separator (_type_, optional): Separator that combines the author and content under one column. Defaults to ': '.

    Returns:
        pd.DataFrame: Dataframe with a column that combines both author and content.
    """
    if out_col is not None:
        df[out_col] = df[author_col] + separator + df[content_col]
    else:
        df[content_col] = df[author_col] + separator + df[content_col]
    return df

def select_columns(df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    """Select columns to keep within the dataframe.

    Args:
        df (pd.DataFrame): Origin of the dataset.
        selected_col (list[str]): Columns to keep inside the dataset.

    Returns:
        pd.DataFrame: Filtered dataset where only the selected columns are kept.
    """
    df = df[selected_cols]
    return df

def balance_binary_classes(majority: pd.DataFrame, minority: pd.DataFrame, upsample: bool = True,
                    shuffle: bool = True, random_state: int | None = None) -> pd.DataFrame:
    """Balances binary classes given the majority label dataframe and minority label dataframe.

    Args:
        majority (pd.DataFrame): Records of the dataset that has the larger label class portion
        minority (pd.DataFrame): Records of the dataset that has the smaller label class portion
        upsample (bool, optional): Determines whether the returned dataset is an upsampled dataset. Defaults to True.
        shuffle (bool, optional): Determines whether the returned dataset is shuffled after balancing. Defaults to True.
        random_state (int | None, optional): Defines the seed used for sampling. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with balanced classes.
    """
    if upsample:
        balanced = pd.concat([
            majority,
            minority.sample(
                len(majority), 
                random_state = random_state, 
                replace = True
            )
        ])
    else:
        balanced = pd.concat([
            majority.sample(
                len(minority), 
                random_state = random_state, 
                replace = True
            ),
            minority,
        ])
    
    if shuffle:
        balanced = balanced.sample(
            frac = 1, 
            random_state = random_state
        ).reset_index(drop = True)
    
    return balanced

def encode_label_texts(df: pd.DataFrame, label_column: str, mapping_dict: dict) -> pd.DataFrame:
    """Encode a label column given a mapping dictionary that encodes a given class or label to an index.

    Args:
        df (pd.DataFrame): The dataframe that contains the label column to be encoded.
        label_column (str): Name of the label column within the provided df dataframe.
        mapping_dict (dict): Class mapping encoding class to an index or number.

    Returns:
        pd.DataFrame: Dataframe with labels encoded according to mapping dictionary.
    """
    df[label_column] = df[label_column].map(mapping_dict)
    return df

def train_tokenizer(doc: str, model_dir: str = '..\\models\\bpe', model_prefix: str = 'spm', 
                    vocab_size: int = 8000, model_type: str = 'bpe', pad_id: int = 3) -> None:
    """Train a SentencePieceModel Tokenizer

    Args:
        doc (str): Path to the document or corpus used to train the tokenizer.
        model_dir (str, optional): Directory to download model and vocabulary. Defaults to ''.
        model_prefix (str, optional): Prefix that spm uses as names to model and vocabulary files. Defaults to 'spm'.
        vocab_size (int, optional): Allowed size of the vocabulary that the tokenizer uses. Defaults to 8000.
        model_type (str, optional): Type of model to be trained using spm's trainer. Defaults to 'bpe'.
        pad_id (int, optional): Index of the padding token. Defaults to 3.
    """
    spm.SentencePieceTrainer.train(
        input = doc,
        model_prefix = os.path.join(model_dir, model_prefix),
        vocab_size = vocab_size,
        model_type = model_type,
        pad_id = pad_id
    )
    return

def encode_feature_texts(df: pd.DataFrame, feature_col: str, model_dir: str = '..\\models\\bpe', 
                         model_prefix: str = 'spm',) -> pd.DataFrame:
    """Encode a feature column within a dataframe using an existing spm model. 
    If no spm model path is provided, an spm model is trained by default.

    Args:
        df (pd.DataFrame): Dataframe that contains the feature column to be encoded.
        feature_col (str): Name of the feature column within the dataframe.
        model_dir (str, optional): Directory to save the model if an spm model is trained. Defaults to '..\\models\\bpe'.
        model_prefix (str, optional): Prefix to use as a name to store the model when it is trained. Defaults to 'spm'.

    Returns:
        pd.DataFrame: Dataframe with encoded feature column using an spm model.
    """
    model_path = os.path.join(model_dir, model_prefix + '.model')
        
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    
    df[feature_col] = df[feature_col].apply(tokenizer.encode)
    
    return df

def process_data(df: pd.DataFrame, feature_col: str, label_col: str, **kwargs) -> pd.DataFrame:
    """Process a classification dataset containing binary classes along with string features and labels
    using a sentencepiece model tokenizer. Keyword arguments can be provided for specific functions such as
    the name of the dataset for dataset-specific actions or corpus path. See other functions for kwargs.

    Args:
        df (pd.DataFrame): Dataframe of the dataset that will be processed.
        feature_col (str): Name of the feature column to be processed inside the df DataFrame.
        label_col (str): Name of the label column to be processed inside the df DataFrame.
        
    Returns:
        pd.DataFrame: Processed dataset with feature and label encoded in numerical representations.
    """
    
    # Kwargs
    df_name = kwargs.get('df_name')
    corpus_path = os.path.join(kwargs.get('dataset_dir','..//datasets'), 'corpus.txt')
    tokenizer_dir = kwargs.get('tokenizer_dir', os.path.join('..','models','bpe'))
    model_prefix = kwargs.get('model_prefix', 'spm')
    vocab_size = kwargs.get('vocab_size', 8000)
    model_type = kwargs.get('model_type', 'bpe')
    pad_id = kwargs.get('pad_id', 3)
    upsample = kwargs.get('upsample',True)
    shuffle = kwargs.get('shuffle', True)
    random_state = kwargs.get('random_state')
    mapping_dict = kwargs.get(
            'mapping_dict',
            {'Credible':0, 'Not Credible': 1}
        )
    test_size = kwargs.get('test_size', 0.2)
    
    # Specific Dataset Processes
    # Combine Author and Content
    if df_name == 'Philippine Fake News Corpus.csv':
        df = combine_author(
            df = df,
            author_col = 'Brand',
            content_col = 'Content'
        )
        
        df = select_columns(
            df = df,
            selected_cols = ['Content','Label']
        )
        
    
    # Balance Classes
    label_dist = df[label_col].value_counts()
    majority = df[df[label_col] == label_dist.index[label_dist.argmax()]]
    minority = df[df[label_col] == label_dist.index[label_dist.argmin()]]
    
    is_imbalanced = len(minority)/len(df) < 0.4
    
    if is_imbalanced:
        balanced_df = balance_binary_classes(
            majority = majority,
            minority = minority,
            upsample = upsample,
            shuffle = shuffle,
            random_state = random_state
        )
              
    # Encode
    encoded_df = encode_label_texts(
        df = balanced_df, 
        label_column = label_col, 
        mapping_dict = mapping_dict
    )
    
    train, test = train_test_split(
        encoded_df, 
        test_size = test_size,
        random_state = random_state
    )
    
    if not os.path.isfile(corpus_path):
        with open(corpus_path, 'w', encoding = 'utf-8') as file:
            for row in train[feature_col]:
                file.write(row + '\n')
    
    
    if not os.path.isfile(os.path.join(tokenizer_dir, model_prefix + '.model')):
        train_tokenizer(
            doc = corpus_path,
            model_dir = tokenizer_dir,
            model_prefix = model_prefix,
            vocab_size = vocab_size,
            model_type = model_type,
            pad_id = pad_id
        )
    
    train = encode_feature_texts(
        df = train,
        feature_col = feature_col,
        model_dir = tokenizer_dir,
        model_prefix = model_prefix,
    ).reset_index(drop=True)
    
    test = encode_feature_texts(
        df = test,
        feature_col = feature_col,
        model_dir = tokenizer_dir,
        model_prefix = model_prefix
    ).reset_index(drop=True)
    
    return train, test