import timeit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
import requests
import joblib
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from sklearn.model_selection import train_test_split
import random
import traceback
warnings.filterwarnings("ignore")

import streamlit as st
df=st.cache_data(pd.read_csv)('https://media.githubusercontent.com/media/AsadNaeem361/fyp/main/creditcard.csv')


#function for best model predictor
def page1():    
    st.info("If you have a file to upload, please use the file uploader (the file should have the same structure as the default). Otherwise, you can continue with the default dataset.")
    uploaded_file = st.file_uploader("Upload Files",type=['csv'], key="fileuploader1")
    if uploaded_file is not None:    
        # Read the CSV data using pandas
        df = pd.read_csv(uploaded_file)
        df=df[['V18','V7','V3','V4','V16','V11','V10','V12','V14','V17','Class']]
        st.write(df)
    else:
        df=st.cache_data(pd.read_csv)('https://media.githubusercontent.com/media/AsadNaeem361/fyp/main/Test_set_25.csv')
        df=df[['V18','V7','V3','V4','V16','V11','V10','V12','V14','V17','Class']]

    # Load the saved model
    model = joblib.load("best_model.joblib")

    st.info('You can select the entire dataset or 100 random rows to feed to the model or input feature values manually')
    option = st.radio("Select", ["Select all rows", "Select 100 random rows", "Input feature values manually"])

    if option == "Select all rows":
        if st.button('Run model'):
            X_test, y_test = df.iloc[:, :-1], df.iloc[:, -1]
            compute_performance2(model, X_test, y_test)

    if option == "Select 100 random rows":
        #100 random records displayed
        X=df.drop(['Class'], axis=1)
        y=df.Class
        rand_df_X = df.sample(n=95, random_state=42)
        rand_df_y = df[df['Class'] == 1].sample(n=5, random_state=42)
        rand_df = pd.concat([rand_df_X, rand_df_y])
        rand_df = rand_df.sample(frac=1, random_state=42)
        if st.button('Run model'):
            st.write("rand_df shape:", rand_df.shape)
            st.write("rand_df contents:", rand_df)
            compute_performance2(model, rand_df.iloc[:, :-1], rand_df.iloc[:, -1])
    
    if option == "Input feature values manually":
        # Create a dictionary to store the input values
        input_dict = {}
        cols = ['V18', 'V7', 'V3', 'V4', 'V16', 'V11', 'V10', 'V12', 'V14', 'V17', 'Class']
        col_dict = {}

        col1, col2, col3 = st.columns(3)

        for i, col in enumerate(cols):
            if i % 3 == 0:
                col_dict[col] = col1
            elif i % 3 == 1:
                col_dict[col] = col2
            else:
                col_dict[col] = col3

        for col in cols[:-1]:
            input_dict[col] = col_dict[col].number_input(f'Enter value for {col}:', step=0.01, key=col)

        # Add a button to fill in remaining values
        if st.button('Check fraud'):
            data = [input_dict]
            X_test_input_dict = pd.DataFrame(data)
            st.write(X_test_input_dict)
            if model.predict(X_test_input_dict)[0] == 1:
                st.error('Fraud Detected')
            else:
                st.success('Valid transaction')
                st.balloons()

# function for build your own model page
def page2():
    st.info("If you have a file to upload, please use the file uploader (the file should have the same structure as the default). Otherwise, you can continue with the default dataset.")
    uploaded_file = st.file_uploader("Upload Files",type=['csv'], key="fileuploader2")
    if uploaded_file is not None:    
        # Read the CSV data using pandas
        df = pd.read_csv(uploaded_file)
        st.write(df)
    else:
        df = st.cache_data(pd.read_csv)('https://media.githubusercontent.com/media/AsadNaeem361/fyp/main/creditcard.csv')
    # Print shape and description of the data
    if st.sidebar.checkbox('Show what the dataframe looks like'):
        st.write(df.head(100))
        st.write('Shape of the dataframe: ',df.shape)
        st.write('Data decription: \n',df.describe())

    if st.sidebar.checkbox('Show fraud and valid transaction details'):
        # Print valid and fraud transactions
        fraud=df[df.Class==1]
        valid=df[df.Class==0]
        outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
        st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
        st.write('Fraud Cases: ',len(fraud))
        st.write('Valid Cases: ',len(valid))
        
    #Obtaining X (features) and y (labels)
    X=df.drop(['Class'], axis=1)
    y=df.Class

    size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 1, shuffle=True, stratify=y)

    # scale column amount and drop column time
    from sklearn.preprocessing import RobustScaler

    rob_scaler = RobustScaler()

    try:
        X_train['scaled_amount'] = rob_scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
        X_train.drop(['Time', 'Amount'], axis=1, inplace=True)

        X_test['scaled_amount'] = rob_scaler.fit_transform(X_test['Amount'].values.reshape(-1, 1))
        X_test.drop(['Time', 'Amount'], axis=1, inplace=True)
    except Exception as e:
        st.error(f"Error: {traceback.format_exc()}")
        
    #Print shape of train and test sets
    if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
        st.write('X_train: ',X_train.shape)
        st.write('y_train: ',y_train.shape)
        st.write('X_test: ',X_test.shape)
        st.write('y_test: ',y_test.shape)
        
    #Import classification models and metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    #initialization of models that performed best
    logreg=LogisticRegression(C=10, max_iter= 15000)
    svm=SVC(kernel='poly', degree=7)
    knn=KNeighborsClassifier(n_neighbors=3)
    rforest=RandomForestClassifier(max_depth= 7, n_estimators= 200)
    xgboost = XGBClassifier(learning_rate= 0.1, max_depth= 7)

    X_train_sfs=X_train
    X_test_sfs=X_test

    X_train_sfs_scaled=X_train_sfs
    X_test_sfs_scaled=X_test_sfs

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
    #Import performance metrics
    from sklearn.metrics import ConfusionMatrixDisplay
    # GridsearchCV
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    np.random.seed(42) #for reproducibility since SMOTE uses randomizations

    smt = SMOTE(sampling_strategy='minority',random_state=0)
    rus = RandomUnderSampler(sampling_strategy='majority',random_state=0)

    alg=['Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression', 'XGBoost']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    rectifier=['SMOTE','RandomUnderSampler','No Rectifier']
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier)

    #Run different classification models with rectifiers
    if st.sidebar.button('Run model'):
        try:
            if classifier=='Logistic Regression':
                model=logreg
            elif classifier == 'k Nearest Neighbor':
                model=knn
            elif classifier == 'Support Vector Machine':
                model=svm
            elif classifier == 'Random Forest':
                model=rforest
            elif classifier == 'XGBoost':
                model=xgboost
            if imb_rect=='No Rectifier':
                compute_performance1(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance1(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='RandomUnderSampler':
                rect=rus
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance1(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        except Exception as e:
            st.error(f"Error: {traceback.format_exc()}")

#for build your own model page
def compute_performance1(model, X_train, y_train,X_test,y_test):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import ConfusionMatrixDisplay
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()
    st.write('F1-Score: ',scores)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Plot confusion matrix
    fig, ax = plt.subplots()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['non-fraudulent', 'fraudulent'])
    cm_display.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)
    cr=classification_report(y_test, y_pred, target_names=['non-fraudulent','fraudulent'])
    st.text('Classification Report: ')
    st.text(cr)
    mcc= matthews_corrcoef(y_test, y_pred)
    st.write('Matthews Correlation Coefficient: ',mcc)
    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for performance computation: %.2f minutes'%(elapsed/60))
    
#for best model predictor page
def compute_performance2(model,X_test,y_test):
    from sklearn.metrics import ConfusionMatrixDisplay
    start_time = timeit.default_timer()
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['non-fraudulent', 'fraudulent'])
    cm_display.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)
    cr=classification_report(y_test, y_pred, target_names=['non-fraudulent','fraudulent'])
    st.text('Classification Report: ')
    st.text(cr)
    mcc= matthews_corrcoef(y_test, y_pred)
    st.write('Matthews Correlation Coefficient: ',mcc)
    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for performance computation: %.2f minutes'%(elapsed/60))