import pandas as pd

def balance_classes(majority: pd.DataFrame, minority: pd.DataFrame, upsample: bool = True,
                    shuffle: bool = True, random_state: int | None = None) -> pd.DataFrame:
    """Balances binary classes given the majority label dataframe and minority label dataframe.

    Args:
        majority (pd.DataFrame): records of the dataset that has the larger label class portion
        minority (pd.DataFrame): records of the dataset that has the smaller label class portion
        upsample (bool, optional): boolean argument that determines whether the returned dataset is an upsampled dataset. Defaults to True.
        shuffle (bool, optional): boolean argument that determines whether the returned dataset is shuffled after balancing. Defaults to True.
        random_state (int | None, optional): integer argument that defines the seed used for sampling. Defaults to None.

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