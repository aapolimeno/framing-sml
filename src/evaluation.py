from sklearn.metrics import classification_report
#import matplotlib.pyplot as plt


def get_report(true_labels, pred_labels, framing): 
    # Calculate precision, recall, and F1-score
    report = classification_report(true_labels, pred_labels, zero_division=1) 
    print(f'-------- Classification report for {framing} framing:')
    print(report)
    
    
def get_confusion_matrix(true_labels, pred_labels, framing): 
    
    emphasis = ["E-responsibility", "E-conflict", "E-economic consequences", "E-human interest",
               "E-morality", "E-info & stats"]

    if framing in emphasis:
        labels = [f'{framing}', 'None']
    
    elif framing == 'valence': 
        labels = ['V-pos', 'V-mod', 'V-neg', 'None']
    
    elif framing == 'risk': 
        labels = ['R-risk', 'R-balanced', 'R-opportunity', 'None']
    
    else: 
        print("Framing type not found, please specify 'framing' variable in main.py")
    
    

    # Get the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels = labels)

    # Display 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap = 'Greens')
    


def start_eval(true_labels, pred_labels, framing, algorithm, text_rep, input_type, df): 
    
    get_report(true_labels, pred_labels, framing)
    #get_confusion_matrix(true_labels, pred_labels, framing, algorithm, text_rep, input_type) 