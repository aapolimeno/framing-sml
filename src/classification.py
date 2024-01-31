from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def convert_labels(test_labels, preds, framing):
    # Convert numerical labels back to strings for readability 
    if framing == 'valence':
        reversed_mapping = {1:'V-pos', 2:'V-mod', 3:'V-neg', 4:'None'}
    if framing == 'risk': 
        reversed_mapping = {1:'R-risk', 2:'R-balanced', 3:'R-opportunity', 4:'None'}
        
    emphasis = ["E-responsibility", "E-conflict", "E-economic consequences", "E-human interest",
               "E-morality", "E-info & stats"]
    
    if framing in emphasis: 
        reversed_mapping = {1:f'{framing}', 0:'None'}

    true_labels = [reversed_mapping.get(item, 4) for item in test_labels]
    pred_labels = [reversed_mapping.get(item, 4) for item in preds]
    
    return true_labels, pred_labels

def apply_svm(train_features, train_labels, test_features, test_labels, framing):
    # Create SVM classifier 
    svm_classifier = SVC(kernel='linear', C=1.0)
    
    #print('-------- Performing 10-fold cross-validation')
    
    # Initialize a KFold cross-validation object
    #kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform five-fold cross-validation
    #cv_scores = cross_val_score(svm_classifier, train_features, train_labels, cv=kf, scoring='accuracy')
    
    # Train the classifier 
    svm_classifier.fit(train_features, train_labels)
    # Predict on the test set 
    svm_preds = svm_classifier.predict(test_features)
    
    # Apply label conversion for readability 
    true_labels, pred_labels = convert_labels(test_labels, svm_preds, framing)

    return true_labels, pred_labels


def apply_sgd(train_features, train_labels, test_features, test_labels, framing):
    """
    TO DO: HYPERPARAMETER TUNING 
    """
    # Create a Stochastic Gradient Descent classifier
    sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    
    #print('-------- Performing 10-fold cross-validation')
    
    # Initialize a KFold cross-validation object
    #kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform five-fold cross-validation
    #cv_scores = cross_val_score(sgd, train_features, train_labels, cv=kf, scoring='accuracy')
    
    # Train the classifier
    sgd.fit(train_features, train_labels)
    # Predict on the test set 
    sgd_preds = sgd.predict(test_features)
    
    # Apply label conversion for readability 
    true_labels, pred_labels = convert_labels(test_labels, sgd_preds, framing)
    
    return true_labels, pred_labels


def apply_lr(train_features, train_labels, test_features, test_labels, framing):

    # Create a Stochastic Gradient Descent classifier
    lr_model = LogisticRegression(max_iter = 4000)
    
    #print('-------- Performing 10-fold cross-validation')
    
    # Initialize a KFold cross-validation object
    #kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform five-fold cross-validation
    #cv_scores = cross_val_score(lr_model, train_features, train_labels, cv=kf, scoring='accuracy')
    
    #print("CROSS VALIDATION RESULTS: ")
    #print(cv_scores)
    
    
    # Scale the features 
    # Train: 
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    
    # Test: 
    scaler = StandardScaler().fit(test_features)
    test_features = scaler.transform(test_features)
    

    # Train the classifier
    lr_model.fit(train_features, train_labels)
    # Predict on the test set 
    lr_preds = lr_model.predict(test_features)
    
    # Apply label conversion for readability 
    true_labels, pred_labels = convert_labels(test_labels, lr_preds, framing)
    
    return true_labels, pred_labels


def apply_pa(train_features, train_labels, test_features, test_labels, framing):

    pac = PassiveAggressiveClassifier(max_iter=2000)
    
    #print('-------- Performing 10-fold cross-validation')
    
    #kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform five-fold cross-validation
    #cv_scores = cross_val_score(pac, train_features, train_labels, cv=kf, scoring='accuracy')
    
    # Train the classifier
    pac.fit(train_features, train_labels)
    # Predict on the test set 
    pac_preds = pac.predict(test_features)
    
    # Apply label conversion for readability 
    true_labels, pred_labels = convert_labels(test_labels, pac_preds, framing)
    
    return true_labels, pred_labels
    
    