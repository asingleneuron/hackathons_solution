
# Import the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For streamlit
import streamlit as st

def problem_description():
    problem_statement = '''
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
    st.markdown(problem_statement)


def graph_demo():
    x = ['Apple', 'mango', 'banana', 'oranges']
    y = [ 3, 10, 6, 20]

    plt.figure(figsize=(10,5))
    sns.barplot(x, y)

    st.pyplot(plt)

@st.cache
def get_input_dataset():
    train_df = pd.read_csv("dataset/Train.csv")
    test_df = pd.read_csv("dataset/Test.csv")

    return train_df, test_df


if __name__ == "__main__":
    print("Main strats from here..")

    # How to set title for the Application
    st.title("Novartis Data Science Challenge...")

    # How to use markdown
    #problem_description()

    # How to add graphs in Streamlit App
    # graph_demo()

    # How to add sidebar and widgets
    st.sidebar.subheader('Features')

    if st.sidebar.checkbox('Problem Description'):
        problem_description()

    # if st.sidebar.checkbox('Show Demo Graph'):
    #     graph_demo()

    # Let's load the data ...
    train_df, test_df = get_input_dataset()

    print(train_df.shape, test_df.shape)

    print(train_df.head())

    # Show dataframe into the streamlit dashboard
    st.write(train_df[:500])


