import streamlit as st
import altair as alt

def render(df, df_importances):
    st.write("### Telco Customer Database")
    st.dataframe(df.head(100), height=150)

    st.write("---")
    st.write("### Model Performance (XGBoost)")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Overall Accuracy", value="76%")
    col2.metric(label="Churn Recall (Caught)", value="78%") 
    col3.metric(label="F1-Score", value="77%")

    st.write("---")
    st.write("### Top Impactful Features")
    chart = alt.Chart(df_importances.head(10)).mark_bar().encode(
        x=alt.X('Importance (%):Q', title='Importance (%)'),
        y=alt.Y('Feature:N', sort='-x', title='Feature', axis=alt.Axis(labelLimit=0, labelFontSize=12, titleFontSize=14)),
        color=alt.Color('Importance (%):Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Feature', 'Importance (%)']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)