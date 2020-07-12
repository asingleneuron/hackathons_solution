import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import perform_chi2_test, benchmark_model, final_customized_model
from collections import defaultdict
import lightgbm as lgb

sns.set_style('darkgrid')

@st.cache
def problem_description():
    problem_statement =   '''
        ### Predict if the server will be hacked
        
        > All the countries across the globe have adapted to means of digital payments. And with the increased volume of digital payments, hacking has become a pretty common event wherein the hacker can try to hack your details just with your phone number linked to your bank account. However, there is a data with some anonymized variables based on which one can predict that the hack is going to happen.
        
        > Your works is to build a predictive model which can identify a pattern in these variables and suggest that a hack is going to happen so that the cyber security can somehow stop it before it actually happens.
        
        > You have to predict the column **"MALICIOUS_OFFENSE"**
        
        
        ### Data:
        
        | Column | Description |
        |--------|:------------|
        | INCIDENT_ID | Unique identifier for an incident log |
        | DATE | Date wof incident occurenc |
        | X_1 - X_15 | Anonymized logging parameters |
        | MULTIPLE_OFFENSE | Target indicates if the incident was a hack |
        
        
        ### Evaluation Matrix:
        
        score = recall_score(actual_valus, predicted_valus, average='macro')
        
        '''

    return problem_statement


# Caching
@st.cache(show_spinner=True, allow_output_mutation=True)
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['DATE'] = pd.to_datetime(data['DATE'])
    return data


# data_load_state = st.text('Loading data...')
train_df = load_data("../Dataset/Train.csv")
test_df = load_data("../Dataset/Test.csv")
# data_load_state.text('Loading data...done!')

bar_plot_map = {}

def create_bar_plot(df, column, target_column=None):
    plt.figure(figsize=(15,10))
    plt.xticks(rotation=90)
    if target_column:
        sns.countplot(column, data=df, hue=target_column)
    else:
        sns.countplot(column, data=df)
    plt.title("Frequency-distribution {}".format(column))
    return plt


def compare_train_test_distribution(X_train, X_test, column):

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    f.suptitle("Train-Test {} distribution".format(column))

    sns.countplot(column, data=X_train, ax=axes1)
    sns.countplot(column, data=X_test, ax=axes2)

    return plt

def missing_value_analysis(X_train, X_test):
    f, (axes1, axes2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes1.set_title('Test-data')
    axes2.set_title('Train-data')
    plt.xticks(rotation=90)
    X_train_null_index , X_train_null_values = X_train.isnull().sum().index, (X_train.isnull().sum() / X_train.shape[0]).values * 100
    X_test_null_index, X_test_null_values = X_test.isnull().sum().index, (X_test.isnull().sum()/X_test.shape[0]).values * 100

    sns.barplot(X_test_null_index, X_test_null_values, ax=axes2)
    sns.barplot(X_train_null_index, X_train_null_values, ax=axes1)

    p= plt.tight_layout()
    return p

@st.cache
def analysis_of_benchmark_model_utils():
    base_features = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13',
                     'X_14', 'X_15']
    evals_result, feature_importance, validation_score = benchmark_model(train_df, base_features, 'MULTIPLE_OFFENSE')
    return evals_result, feature_importance, validation_score

def analysis_of_benchmark_model():
    evals_result, feature_importance, validation_score = analysis_of_benchmark_model_utils()
    st.subheader("Validation data  recall_score('average'='macro') is : {}".format( round(validation_score,4)))

    plt.figure(figsize=(10, 5))
    plt.title('Feature Importance Plot')
    sns.barplot(feature_importance['Feature Importance'], feature_importance.index)
    p1 = plt
    st.pyplot(p1)

    plt.figure(figsize=(10, 5))
    plt.title('Recall Curve')
    sns.lineplot(x='iterations', y='recall_score', data=evals_result, hue='Recall')  # , ax=axes1)
    p2 = plt
    st.pyplot()

    plt.title('Binary Logloss Curve')
    sns.lineplot(x='iterations', y='binarylogloss_score', data=evals_result, hue='Binary Logloss')  # , ax=axes2)
    st.pyplot()

@st.cache
def analysis_of_final_customized_model_util():
    evals_result, feature_importance, validation_score = final_customized_model(train_df, test_df)
    return evals_result, feature_importance, validation_score

def analysis_of_final_customized_model():
    evals_result, feature_importance, validation_score = analysis_of_final_customized_model_util()
    st.subheader("Validation data  recall_score('average'='macro') is : {}".format( round(validation_score,4)))

    plt.figure(figsize=(10, 5))
    plt.title('Feature Importance Plot')
    sns.barplot(feature_importance['Feature Importance'], feature_importance.index)
    p = plt
    st.pyplot(p, use_container_width=True)

    plt.figure(figsize=(10, 5))
    plt.title('Recall Curve')
    sns.lineplot(x='iterations', y='recall_score', data=evals_result, hue='Recall')  # , ax=axes1)
    st.pyplot()

    plt.title('Binary Logloss Curve')
    sns.lineplot(x='iterations', y='binarylogloss_score', data=evals_result, hue='Binary Logloss')  # , ax=axes2)
    st.pyplot()


def problem_desc_checkbox():
    if st.sidebar.checkbox('Problem Description', True):
       st.markdown(problem_description())

@st.cache
def sample_train_data():
    return train_df[:30]

@st.cache
def sample_test_data():
    return test_df[:30]

def significance_test():
    tmp = defaultdict(list)
    target_column = 'MULTIPLE_OFFENSE'
    base_features = ['X_1',
                     'X_2',
                     'X_3',
                     'X_4',
                     'X_5',
                     'X_6',
                     'X_7',
                     'X_8',
                     'X_9',
                     'X_10',
                     'X_11',
                     'X_12',
                     'X_13',
                     'X_14',
                     'X_15']

    for col in base_features[:]:
        isSignificant, alpha, p_value = perform_chi2_test(train_df, col, target_column)
        tmp['Feature'].append(col)
        tmp['Alpha'].append(alpha)
        tmp['P-Value'].append(p_value)
        tmp['Reject H0 (null hypothesis)'].append(str(isSignificant))

    return tmp

missing_info = missing_value_analysis(train_df, test_df)

def show_eta():
    st.sidebar.subheader("Features")

    problem_desc_checkbox()


    if st.sidebar.checkbox('Train Data'):
        st.subheader('Train-Data')
        st.write(sample_train_data())

    if st.sidebar.checkbox('Test Data'):
        st.subheader('Test-Data')
        st.write(sample_test_data())

    if st.sidebar.checkbox('Missing-Values'):
        st.subheader('Missing-Values Analysis')
        st.pyplot(missing_info)

    target_column = 'MULTIPLE_OFFENSE'
    if st.sidebar.checkbox('Show freq_distribution'):
        option = st.sidebar.selectbox('Select Feature',
                                      ['MULTIPLE_OFFENSE',
                                       'X_1',
                                       'X_2',
                                       'X_3',
                                       'X_4',
                                       'X_5',
                                       'X_6',
                                       'X_7',
                                       'X_8',
                                       'X_9',
                                       'X_10',
                                       'X_11',
                                       'X_12',
                                       'X_13',
                                       'X_14',
                                       'X_15'])

        st.subheader('Frequency distribution')
        if option != target_column:
            st.pyplot(create_bar_plot(train_df, option, target_column), use_container_width=True)
        else:
            st.pyplot(create_bar_plot(train_df, option), use_container_width=True)

    if st.sidebar.checkbox('Compare train-test distribution'):
        option = st.sidebar.selectbox('Select Feature', ['X_1',
                                                         'X_2',
                                                         'X_3',
                                                         'X_4',
                                                         'X_5',
                                                         'X_6',
                                                         'X_7',
                                                         'X_8',
                                                         'X_9',
                                                         'X_10',
                                                         'X_11',
                                                         'X_12',
                                                         'X_13',
                                                         'X_14',
                                                         'X_15'])

        st.subheader('Train-Test data distribution')
        st.pyplot(compare_train_test_distribution(train_df, test_df, option))

    if st.sidebar.checkbox('Do significance-test'):
        st.subheader('Categorical features Significance-Test')
        st.subheader('H0 (null hypothesis) is : there is no dependency between features and target')

        tmp = significance_test()
        st.write(pd.DataFrame(tmp))

    if st.sidebar.checkbox('Analysis of Benchmark Model'):
        st.subheader('Analysis of Benchmark Model')
        analysis_of_benchmark_model()

    if st.sidebar.checkbox('Analysis of Final Customized Model'):
        st.subheader('Final Customized Model')
        analysis_of_final_customized_model()

    if st.sidebar.checkbox('Test-Score'):
        st.subheader('Test-Score {} < used final_customized_model >'.format(100))
        st.subheader("")
        st.image("./test_score.png", use_column_width=True)

def main():
    st.title('Novartis-Data-Science-Challenge')
    st.subheader("")
    show_eta()

if __name__ == "__main__":
    print(st.__version__)
    main()
