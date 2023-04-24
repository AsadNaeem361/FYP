python functions

list(), The list() function creates a list object. A list object is a collection which is ordered and changeable.

zip(), The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc.

lambda(), A lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression.

sorted(), The sorted() function returns a sorted list of the specified iterable object. You can specify ascending or descending order. Strings are sorted alphabetically, and numbers are sorted numerically. Note: You cannot sort a list that contains BOTH string values AND numeric values.




=================================

timeit library

this library is used to measure the execution times of small and large code snippets.

default_timer(), This function returns the waiting time along with the CPU time, and it’s dependent on the platform. Its value would be different for Windows, Linux, etc. Other programs running on the same computer may affect the output time.



===================================

pandas library as pd

pandas is a software library written for the Python programming language 
for data manipulation and analysis. In particular, it offers data structures 
and operations for manipulating numerical tables and time series.

read_csv(), reads a csv

pd.DataFrame.head(), used to get first n rows

pd.DataFrame.describe(),
The describe() method returns description of the data in the DataFrame.
If the DataFrame contains numerical data, the description contains these information for each column:
count - The number of not-empty values.
mean - The average (mean) value.
std - The standard deviation.
min - the minimum value.
25% - The 25% percentile*.
50% - The 50% percentile*.
75% - The 75% percentile*.
max - the maximum value.

pd.DataFrame.attribute.value_counts(), The value_counts() function is used to get a Series containing counts of unique values.

pd.DataFrame.drop(), The drop() method removes the specified row or column.




====================================
pyplot matplotlib library

Pyplot is a Matplotlib module which provides a MATLAB-like interface. Matplotlib is designed to be as usable as MATLAB, with the ability to use Python and the advantage of being free and open-source. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc. The various plots we can utilize using Pyplot are Line Plot, Histogram, Scatter, 3D Plot, Image, Contour, and Polar.

plt.bar(), bar plot

plt.title(), title of plot

plt.xlabel(), plt.ylabel(), x and y axis labels respectively


=====================================

plotly.express library

The plotly.express module (usually imported as px) contains functions that can create entire figures at once, and is referred to as Plotly Express or PX. Plotly Express is a built-in part of the plotly library, and is the recommended starting point for creating most common figures.

=====================================

numpy library 

NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

np.bincount(), The np.bincount() is a numpy library method used to obtain the frequency of each element provided inside a numpy array. The numpy bincount() method takes arr_name, weights, and minlength as arguments and returns the ndarray of integers.

=====================================

warnings library

Warnings are provided to warn the developer of situations that aren’t necessarily exceptions. Usually, a warning occurs when there is some obsolete of certain programming elements, such as keyword, function or class, etc. A warning in a program is distinct from an error. Python program terminates immediately if an error occurs. Conversely, a warning is not critical. It shows some message, but the program runs. The warn() function defined in the ‘warning‘ module is used to show warning messages. The warning module is actually a subclass of Exception which is a built-in class in Python.

=======================================

streamlit library

Streamlit is a free and open-source framework to rapidly build and share beautiful machine learning and data science web apps. It is a Python-based library specifically designed for machine learning engineers.


st.title(), Display text in title formatting.

st.cache(), if function is executed with same parameter values, the return value is just taken from the cache instead of calculating again.

st.write(), Write arguments to the app. write() has some unique properties:
You can pass in multiple arguments, all of which will be written.
Its behavior depends on the input types as follows.
It returns None, so its "slot" in the App cannot be reused.

st.sidebar.checkbox(), creates checkbox in sidebar

st.sidebar.slider(), creates slider bar in sidebar

st.sidebar.selectbox(), creates dropdown in sidebar

st.pyplot(), Display a matplotlib.pyplot figure.




======================================

sklearn library

Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software machine learning library for the Python programming language.[3] It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.


train_test_split(), The train_test_split function of the sklearn.model_selection package in Python splits arrays or matrices into random subsets for train and test data, respectively.

Models:
LogisticRegression()
SVC()
KNeighborsClassifier()
ExtraTreesClassifier(random_state=42)
RandomForestClassifier(random_state=42)

model.feature_importances_, The importance of a feature is basically: how much this feature is used in each tree of the forest. Formally, it is computed as the (normalized) total reduction of the criterion brought by that feature.

cross_val_score(), The cross_val_score() function will be used to perform the evaluation, taking the dataset and cross-validation configuration and returning a list of scores calculated for each fold.

model.fit(),  The ‘fit’ method trains the algorithm on the training data, after the model is initialized. 

model.predict(), predict() will perform predictions on the testing instances, based on the learned parameters during fit


=======================================
imblearn library

SMOTE(), Class to perform over-sampling using SMOTE.

NearMiss(), What is the Near-Miss Algorithm? Near-miss is an algorithm that can help in balancing an imbalanced dataset. It can be grouped under undersampling algorithms and is an efficient way to balance the data. The algorithm does this by looking at the class distribution and randomly eliminating samples from the larger class.


