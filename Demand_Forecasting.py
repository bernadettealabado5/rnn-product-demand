import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    df1 = filtered_df.copy()
    df1 = df1.loc[:, ['date', 'sales']]
    df1['date'] = pd.to_datetime(df1['date'])

    # Set the 'date' column as the index
    df1.set_index('date', inplace=True)

    st.write("The TIme Series Dataset")
    st.write(df1)   
    st.erite(df1.shape)
    
    st.write("The Time Series Plot")

    # Assuming your dataframe is called 'df'
    fig, ax = plt.subplots()  # Create a figure and an axes

    # Plot the timeseries data on the axes
    ax.plot(df1['sales'])

    # Optional customizations
    ax.set_title('Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.grid(True)  # Add gridlines for better readability

    # Limit the number of ticks on the x-axis to 10 (adjust as needed)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability (optional)
    locator = plt.MaxNLocator(nbins=10)
    ax.xaxis.set_major_locator(locator)
    st.pyplot(fig)



if __name__ == "__main__":
    app()   
