from preprocessing import get_train_test
from text_representation import get_repr
from classification import apply_lr, apply_sgd
from evaluation import start_eval
import pandas as pd

# LOAD DATA

path = 'data/processed/annotated_sample.csv'
df = pd.read_csv(path, sep=';')

if __name__ == '__main__':
    # To set framing variable, choose from the following options:
    framing = ['risk', 'valence', 'E-responsibility', 'E-morality',
           'E-conflict','E-economic consequences',
           'E-human interest', 'E-info & stats']
    
    # Text representation options: 
    text_reps = ['tfidf']

    # Classification/algorithm options: 
    algorithms = ['logreg']
    #algorithms = ['svm', 'logreg', 'pac']

    # Input type
    # choose from: 'texts', 'titles'
    input_type = 'texts'

    #if input_type == 'paragraphs':

    for text_rep in text_reps: 
        
        # Get text representations 
        print(f'--- Starting the pipeline for {text_rep} representations')
        #df = get_tfidf(df)
        df = get_repr(df, text_rep, input_type) 
    
        for frame in framing: 
            print(f'----- Working on {frame} framing')
            
            # Split data into train + test, get features and labels 
            train_features, train_labels, test_features, test_labels = get_train_test(df, frame, text_rep, input_type)
            
            for algorithm in algorithms: 
    
                # Apply classification
                print(f'-------- Training a {algorithm} classifier')

                if algorithm == 'sgd':
                    true_labels, pred_labels = apply_sgd(train_features, 
                        train_labels, test_features, test_labels, frame)
                    # Perform evaluation 
                    start_eval(true_labels, pred_labels, frame, algorithm, text_rep, input_type, df)
                    
                elif algorithm == 'logreg': 
                    true_labels, pred_labels = apply_lr(train_features, 
                        train_labels, test_features, test_labels, frame) 
                    # Perform evaluation 
                    start_eval(true_labels, pred_labels, frame, algorithm, text_rep, input_type, df)
                    
                else: 
                    print("ALGORITHM NOT FOUND: check the code for typos")
                    
            print('\n')
            print('===============================')
            print(f'----- Done with {frame}')
            print('===============================')
            print('\n')

    print('----- Done!!!')
