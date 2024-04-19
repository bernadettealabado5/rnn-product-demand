import streamlit as st
import pandas as pd

def app():
    st.title('Demand Forecasting')
    st.write('Welcome to the Demand Forecasting page!') 

    df = pd.read_csv('./data/train.csv')

    # Filter records based on store and item (assuming single values)
    options = df['store'].unique()
    # Create the option box using st.selectbox
    selected_option = st.sidebar.selectbox("Select a store:", options)
    store_to_filter = selected_option
    options = df['item'].unique()
    # Create the option box using st.selectbox
    selected_option = st.sidebar.selectbox("Select item:", options)
    item_to_filter = selected_option

    filtered_df = df[(df['store'] == store_to_filter) & (df['item'] == item_to_filter)]

    st.write("\nFiltered data:")
    st.write(filtered_df)
    st.write(filtered_df.shape)

if __name__ == "__main__":
    app()   
