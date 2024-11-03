import pandas as pd
import streamlit as st
from utils import ImprovedCBR

train_data = pd.read_csv("Dataset/Train_cbr_case_binary.csv", index_col=0)
cbr_obj = ImprovedCBR(train_data)

st.subheader("Evaluate CBR Performance on Test Set")

container = st.container(border=True)
uploaded_test_file = container.file_uploader("Upload the test file containing symptoms and diseases", type=["csv"])
submit = container.button("Evaluate Performace", type="primary")

if submit:
    if uploaded_test_file is not None:
        test_df = pd.read_csv(uploaded_test_file, index_col=0)
        test_set_problems = test_df[test_df.columns[:-2]]
        test_set_actual_solutions = test_df[test_df.columns[-2:]]
        predicted_solutions = cbr_obj.get_best_solutions_for_test_cases(test_set_problems)
        y_true = test_set_actual_solutions[test_set_actual_solutions.columns[0]]
        y_pred = predicted_solutions[predicted_solutions.columns[0]]
        metrics = cbr_obj.evaluate_cbr_metrics(true_values=y_true, predicted_values=y_pred)
        fig = cbr_obj.visualize_metrics(metrics=metrics)
        container2 = st.container(border=True)
        container2.write("""<h2 style="text-align: center"> Evaluation Results </h2>""", unsafe_allow_html=True)
        container2.write("""
        ---
        """)
        container2a = container2.container(border=True)
        col1, col2 = container2a.columns([2,2])
        col1.write("#### Predicted Solutions")
        col1.write(predicted_solutions)

        col2.write("#### Actual Solutions")
        col2.write(test_set_actual_solutions)

        container2b = container2.container(border=True)
        container2b.write("#### Metrics")
        container2b.write(metrics)
        
        container2c = container2.container(border=True)
        container2c.write("#### Metrics Visualized")
        container2c.pyplot(cbr_obj.visualize_metrics(metrics=metrics))


