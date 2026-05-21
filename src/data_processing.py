import pandas as pd
import sentencepiece as spm
import os

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
        pd.DataFrame: dataframe with balanced classes.
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
    df[label_column] = df[label_column].apply(mapping_dict)
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

def encode_feature_texts(df: pd.DataFrame, feature_col: str, model_path: str | None = None,
                         dataset_dir: str = '..\\datasets', model_dir: str = '..\\models\\bpe', 
                         model_prefix: str = 'spm', vocab_size: int | None = None, 
                         model_type: str = 'bpe', pad_id: int = 3) -> pd.DataFrame:
    """Encode a feature column within a dataframe using an existing spm model. 
    If no spm model path is provided, an spm model is trained by default.

    Args:
        df (pd.DataFrame): Dataframe that contains the feature column to be encoded.
        feature_col (str): Name of the feature column within the dataframe.
        model_path (str | None, optional): Path to the trained spm model tokenizer. If None, trains an spm model tokenizer. Defaults to None.
        dataset_dir (str, optional): Directory to store the corpus if an spm model is trained. Defaults to '..\\datasets'.
        model_dir (str, optional): Directory to save the model if an spm model is trained. Defaults to '..\\models\\bpe'.
        model_prefix (str, optional): Prefix to use as a name to store the model when it is trained. Defaults to 'spm'.
        vocab_size (int | None, optional): Allowed vocabulary size for the model to use if it is trained. Defaults to None.
        model_type (str, optional): Type of tokenizer the model uses when it is trained. Defaults to 'bpe'.
        pad_id (int, optional): Pad Id used by the tokenizer when a model is trained. Defaults to 3.

    Returns:
        pd.DataFrame: Dataframe with encoded feature column using an spm model.
    """
    if model_path is None:
        corpus_path = os.path.join(dataset_dir, 'corpus.txt')
        with open(corpus_path, 'w', encoding = 'utf-8') as file:
            for row in df[feature_col]:
                file.write(row + '\n')
        
        train_tokenizer(
            doc = corpus_path,
            model_dir = model_dir,
            model_prefix = model_prefix,
            vocab_size = vocab_size,
            model_type = model_type,
            pad_id = pad_id
        )
        
        model_path = os.path.join(model_dir, model_prefix + '.model')
        
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    
    df[feature_col] = df[feature_col].apply(tokenizer.encode)
    
    return df