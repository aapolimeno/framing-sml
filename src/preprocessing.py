import pandas as pd 
from sklearn.model_selection import train_test_split


def preprocess_general(data): 
    
    """
    This function performs necessary preprocessing on the general dataset. 
    The following steps are included: 
         - selection of relevant columns 
         - label conversion, where strings containing all labels are transformed into columns 
             containing one label each, with a binary encoding to show if it is present
    
    """
    
    # Select the relevant columns 
    df = data[["id", "text", "label"]]
    
    # Rewrite the labels to binary encodings
    # So transform the label form "E-conflict#R-balanced" to columns with 1/0 for each label
    annotations = df["label"].tolist()

    # The binary representation for each row will take the form of this list: 
    options = ["E-responsibility", "E-conflict", "E-economic consequences", "E-human interest",
               "E-morality", "E-info & stats", "V-positive", "V-moderate", "V-negative",
               "R-balanced", "R-opportunity", "R-risk", "N/A"]
    # Where a 1 will indicate it's present, and a 0 means it's not present. 

    # Contains the binary representation of the labels
    all_coding = []

    # Split the labels on '#', add them to a list 
    all_labels = [str(line).split("#") for line in annotations]

    # This line creates a list of len(options) for each row, 
    # where a 1 is given if the option label is present in the current row, otherwise it gives a 0
    all_coding = [[1 if option in row else 0 for option in options] for row in all_labels]

    
    # Make a new dataframe with only the binary coding
    df = pd.DataFrame(all_coding)

    # Grab the relevant columns from the complete dataframe
    df["article_id"] = data["id"].tolist()
    df["text"] = data["text"].tolist()

    # Rename columns
    df = df.rename(columns={0:"E-responsibility", 1:"E-conflict", 2:"E-economic consequences", 3:"E-human interest",
                            4:"E-morality", 5:"E-info & stats", 6:"V-pos", 7:"V-mod", 8:"V-neg", 9:"R-balanced",
                            10:"R-opportunity", 11:"R-risk", 12:"N/A"})

    
    return df 
    
    
def get_features_and_labels(train, test, framing, text_method): 
    
    emphasis = ["E-responsibility", "E-conflict", "E-economic consequences", "E-human interest",
               "E-morality", "E-info & stats"]
    
    if framing in emphasis:
        var1 = framing
    
    
    elif framing == 'valence': 
        var1 = 'V-pos'
        var2 = 'V-mod'
        var3 = 'V-neg'
        
    elif framing == 'risk': 
        var1 = 'R-risk'
        var2 = 'R-balanced'
        var3 = 'R-opportunity'
        
        
    ### Training features + labels

    # Features
    train_features = train[f"{text_method}"].tolist()
    
    
    emphasis = ["E-responsibility", "E-conflict", "E-economic consequences", "E-human interest",
               "E-morality", "E-info & stats"]
    
    if framing in emphasis: 
        # Labels
        df_labels_train = train[[var1, 'None']]
        train_labels_as_strings = df_labels_train.idxmax(axis=1).tolist()
        
        # Map labels to numerical value
        mapping = {var1:1,  'None':0}
        train_labels = [mapping.get(item, 2) for item in train_labels_as_strings]
        
        ### Testing features + labels 
        # Features
        test_features = test[f"{text_method}"].tolist()

        # Labels
        df_labels_test = test[[var1, 'None']]
        test_labels_as_strings = df_labels_test.idxmax(axis=1).tolist()

        # Map labels to numerical value
        mapping = {var1:1, 'None':0}
        test_labels = [mapping.get(item, 2) for item in test_labels_as_strings]
    
    
    elif framing == 'risk' or framing == 'valence':
        # Labels
        df_labels_train = train[[var1, var2, var3, 'None']]

        train_labels_as_strings = df_labels_train.idxmax(axis=1).tolist()
        
        # Map labels to numerical value
        mapping = {var1:1, var2:2, var3:3, 'None':4}
        train_labels = [mapping.get(item, 4) for item in train_labels_as_strings]

        ### Testing features + labels 
        # Features
        test_features = test[f"{text_method}"].tolist()

        # Labels
        df_labels_test = test[[var1, var2, var3, 'None']]
        test_labels_as_strings = df_labels_test.idxmax(axis=1).tolist()

        # Map labels to numerical value
        mapping = {var1:1, var2:2, var3:3, 'None':4}
        test_labels = [mapping.get(item, 4) for item in test_labels_as_strings]
    
    
    else: 
        print('\n')
        print('===============================================================')
        print("ERROR: framing type not found.")
        print("You may have made a typo when specifying the framing variable.")
        print('===============================================================')
        print('\n')

    
    return train_features, train_labels, test_features, test_labels

    
def get_train_test(df, framing, text_method, input_type): 
    
    if framing == 'valence': 
        # Grab relevant valence columns
        df = df[["article_id", f"{text_method}","V-pos", "V-mod", "V-neg"]]

        # Add None variable if no valence annotation is present
        condition = (df['V-pos'] == 0) & (df['V-mod'] == 0) & (df['V-neg'] == 0)
        df.insert(0, 'None', 0)
        #df["None"] = 0
        df.loc[condition, 'None'] = 1
        
    
    if framing == 'risk': 
        # Grab relevant risk columns
        df = df[['article_id', f'{text_method}', 'R-risk', 'R-balanced', 'R-opportunity']]

        # Add None variable if no risk annotation is present
        condition = (df['R-risk'] == 0) & (df['R-balanced'] == 0) & (df['R-opportunity'] == 0)
        df.insert(0, 'None', 0)
        df.loc[condition, 'None'] = 1

    
    
    emphasis = ["E-responsibility", "E-conflict", "E-economic consequences", "E-human interest",
               "E-morality", "E-info & stats"]
    
    if framing in emphasis: 
        df = df[['article_id', f'{text_method}', f'{framing}']]
        
        # Add None variable
        condition = (df[f'{framing}'] == 0) 
        df.insert(0, 'None', 0)
        df.loc[condition, 'None'] = 1

        
    # Train/test split 
    train, test = train_test_split(df, test_size = 0.20)
        
    train_features, train_labels, test_features, test_labels = get_features_and_labels(train, test, framing, text_method)
        
    return train_features, train_labels, test_features, test_labels



#def augment_data(documents): 
    