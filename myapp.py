import timeit
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
import requests
import joblib
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
import random
import traceback


warnings.filterwarnings("ignore")

import streamlit as st
# df=st.cache(pd.read_csv)('https://media.githubusercontent.com/media/AsadNaeem361/myapp-heroku/main/creditcard.csv')
df=st.cache_data(pd.read_csv)('creditcard.csv')

def page2():
    uploaded_file = st.file_uploader("Upload Files",type=['csv'], key="fileuploader2")
    if uploaded_file is not None:    
        # Read the CSV data using pandas
        df = pd.read_csv(uploaded_file)
    else:
        df = st.cache_data(pd.read_csv)('C:/users/asadn/desktop/FYP/myapp-heroku/creditcard.csv')
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

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split

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
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    logreg=LogisticRegression()
    svm=SVC()
    knn=KNeighborsClassifier()
    rforest=RandomForestClassifier(random_state=42)
    xgboost = XGBClassifier(random_state=42)

    # features=X_train.columns.tolist()


    # #Feature selection through feature importance
    # @st.cache_resource
    # def feature_sort(_model,X_train,y_train):
    #     #feature selection
    #     mod=model
    #     # fit the model
    #     mod.fit(X_train, y_train)
    #     # get importance
    #     imp = mod.feature_importances_
    #     return imp

    # Classifiers for feature importance
    # clf=['Extra Trees','Random Forest']
    # clf=['Random Forest']
    # mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

    # start_time = timeit.default_timer()
    # # if mod_feature=='Extra Trees':
    # #     model=etree
    # #     importance=feature_sort(model,X_train,y_train)
    # if mod_feature=='Random Forest':
    #     model=rforest
    #     importance=feature_sort(model,X_train,y_train)
    # elapsed = timeit.default_timer() - start_time
    # st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))    

    # #Plot of feature importance
    # if st.sidebar.checkbox('Show plot of feature importance'):
    #     fig, ax = plt.subplots()
    #     plt.bar([x for x in range(len(importance))], importance)
    #     plt.title('Feature Importance')
    #     plt.xlabel('Feature (Variable Number)')
    #     plt.ylabel('Importance')
    #     st.pyplot(fig)

    # feature_imp=list(zip(features,importance))
    # feature_sort=sorted(feature_imp, key = lambda x: x[1])

    # n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

    # top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

    # if st.sidebar.checkbox('Show selected top features'):
    #     st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

    # X_train_sfs=X_train[top_features]
    # X_test_sfs=X_test[top_features]


    X_train_sfs=X_train
    X_test_sfs=X_test

    X_train_sfs_scaled=X_train_sfs
    X_test_sfs_scaled=X_test_sfs

    #Import performance metrics, imbalanced rectifiers
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
    #Import performance metrics
    from sklearn.metrics import ConfusionMatrixDisplay
    # GridsearchCV
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    from imblearn.over_sampling import SMOTE
    # from imblearn.under_sampling import NearMiss
    from imblearn.under_sampling import RandomUnderSampler
    np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations

    smt = SMOTE(sampling_strategy='minority',random_state=0)
    # nr = NearMiss()
    rus = RandomUnderSampler(sampling_strategy='majority',random_state=0)

    alg=['Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression', 'XGBoost']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    # rectifier=['SMOTE','Near Miss','RandomUnderSampler','No Rectifier']
    rectifier=['SMOTE','RandomUnderSampler','No Rectifier']
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier)

    #Run different classification models with rectifiers
    if st.sidebar.button('Run credit card fraud detection'):
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
    

def compute_performance2(model,X_test,y_test):
    start_time = timeit.default_timer()
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    st.text('Confusion Matrix: ')
    st.text(cm)
    cr=classification_report(y_test, y_pred, target_names=['non-fraudulent','fraudulent'])
    st.text('Classification Report: ')
    st.text(cr)
    mcc= matthews_corrcoef(y_test, y_pred)
    st.write('Matthews Correlation Coefficient: ',mcc)
    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for performance computation: %.2f minutes'%(elapsed/60))

def page1():    

    uploaded_file = st.file_uploader("Upload Files",type=['csv'], key="fileuploader1")
    if uploaded_file is not None:    
        # Read the CSV data using pandas
        df = pd.read_csv(uploaded_file)
        st.write(df)
    else:
        df=st.cache_data(pd.read_csv)('Test_set_25.csv')


    # Load the saved model
    model = joblib.load("C:/users/asadn/desktop/model.joblib")

    st.write('You can select the entire dataset or 100 random rows to feed to the model')
    option = st.radio("Select", ["Select all rows", "Select 100 random rows", "Input manually values of features"])

    if option == "Select all rows":
        if st.button('Run model'):
            X_test, y_test = df.iloc[:, :-1], df.iloc[:, -1]
            compute_performance2(model, X_test, y_test)

    if option == "Select 100 random rows":
        #100 random records displayed
        rand_df=df.sample(n=100)
        if st.button('Run model'):
            st.write("rand_df shape:", rand_df.shape)
            st.write("rand_df contents:", rand_df)
            X_test, y_test = rand_df.iloc[:, :-1], rand_df.iloc[:, -1]
            compute_performance2(model, X_test, y_test)
    
    if option == "Input manually values of features":
        # Create a dictionary to store the input values
        input_dict = {}
        new_dict = {}

        # Ask the user to input values for features v1-v28
        col1, col2, col3 = st.columns(3)
        for i in range(1, 29, 3):
            input_dict[f'v{i}'] = col1.number_input(f'Enter value for v{i}:', min_value=-1.0, max_value=1.0, step=0.01, key=f'v{i}')
            if i+1 <= 28:
                input_dict[f'v{i+1}'] = col2.number_input(f'Enter value for v{i+1}:', min_value=-1.0, max_value=1.0, step=0.01, key=f'v{i+1}')
            if i+2 <= 28:
                input_dict[f'v{i+2}'] = col3.number_input(f'Enter value for v{i+2}:', min_value=-1.0, max_value=1.0, step=0.01, key=f'v{i+2}')
 
        # Add a button to fill in remaining values
        if st.button('Run the prediction'):
            data = [input_dict]
            X_test_input_dict = pd.DataFrame(data)
            X_test_input_dict['amount'] = 2.22
            X_test_input_dict['time'] = 2.22
            if model.predict(X_test_input_dict)[0] == 1:
                st.error('Fraud Detected')
                st.balloons()
            else:
                st.write('valid transaction')
                st.balloons()

        # Display the input values to the user
        st.write('Input values:')
        st.write(input_dict)
