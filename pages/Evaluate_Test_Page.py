import pandas as pd
import streamlit as st
from utils import ImprovedCBR

train_data = pd.read_csv("Dataset/Train_cbr_case_binary.csv", index_col=0)
cbr_obj = ImprovedCBR(train_data)

st.subheader("Evaluate CBR Performance on Test Set")

container = st.container(border=True)
uploaded_test_file = container.file_uploader("Upload the test file containing symptoms and diseases", type=["csv"])
selected_similarity_metrics = container.selectbox(
    "Select Similarity Metrics",
    options=["hybrid", "cosine", "jaccard", "euclidean", "all metrics"],
    index=0
)
submit = container.button("Evaluate Performace", type="primary")

if submit:
    if uploaded_test_file is not None:
        test_df = pd.read_csv(uploaded_test_file, index_col=0)
        test_set_problems = test_df[test_df.columns[:-2]]
        test_set_actual_solutions = test_df[test_df.columns[-2:]]
        if selected_similarity_metrics != "all metrics":
            predicted_solutions = cbr_obj.get_best_solutions_for_test_cases(
                test_set_problems, similarity_metrics=selected_similarity_metrics)
            y_true = test_set_actual_solutions[test_set_actual_solutions.columns[0]]
            y_pred = predicted_solutions[predicted_solutions.columns[0]]
            metrics = cbr_obj.evaluate_cbr_metrics(
                true_values=y_true,
                predicted_values=y_pred,
                col_label=selected_similarity_metrics
            )
            fig = cbr_obj.visualize_metrics(metrics=metrics, col_label=selected_similarity_metrics)
        else:
            y_true = test_set_actual_solutions[test_set_actual_solutions.columns[0]]

            predicted_solutions = cbr_obj.get_best_solutions_for_test_cases(
                test_set_problems, similarity_metrics="euclidean")
            y_pred = predicted_solutions[predicted_solutions.columns[0]]
            metrics1 = cbr_obj.evaluate_cbr_metrics(
                true_values=y_true,
                predicted_values=y_pred,
                col_label="euclidean"
            )
            
            predicted_solutions = cbr_obj.get_best_solutions_for_test_cases(
                test_set_problems, similarity_metrics="cosine")
            y_pred = predicted_solutions[predicted_solutions.columns[0]]
            metrics2 = cbr_obj.evaluate_cbr_metrics(
                true_values=y_true,
                predicted_values=y_pred,
                col_label="cosine"
            )

            predicted_solutions = cbr_obj.get_best_solutions_for_test_cases(
                test_set_problems, similarity_metrics="jaccard")
            y_pred = predicted_solutions[predicted_solutions.columns[0]]
            metrics3 = cbr_obj.evaluate_cbr_metrics(
                true_values=y_true,
                predicted_values=y_pred,
                col_label="jaccard"
            )

            predicted_solutions = cbr_obj.get_best_solutions_for_test_cases(
                test_set_problems, similarity_metrics="hybrid")
            y_pred = predicted_solutions[predicted_solutions.columns[0]]
            metrics4 = cbr_obj.evaluate_cbr_metrics(
                true_values=y_true,
                predicted_values=y_pred,
                col_label="hybrid"
            )

            metrics = pd.concat([metrics1, metrics2, metrics3, metrics4], axis='columns')

        container2 = st.container(border=True)
        metrics_name = selected_similarity_metrics.capitalize()
        container2.write(
            f"""<h2 style="text-align: center"> Evaluation Results using {metrics_name} Similarity Metrics </h2>""",
            unsafe_allow_html=True
        )
        container2.write("""
        ---
        """)
        container2a = container2.container(border=True)
        col1, col2 = container2a.columns([2,2])
        col1.write(f"#### Predicted Solutions - {selected_similarity_metrics}")
        col1.write(predicted_solutions)

        col2.write("#### Actual Solutions")
        col2.write(test_set_actual_solutions)

        container2b = container2.container(border=True)
        container2b.write(f"#### Metrics - {selected_similarity_metrics}")
        container2b.write(metrics)
        
        container2c = container2.container(border=True)
        container2c.write(f"#### Metrics Visualized - {selected_similarity_metrics}")
        if selected_similarity_metrics == "all metrics":
            # container2c.pyplot(metrics.plot.bar())
            container2c.bar_chart(data=metrics, stack=False, horizontal=True)
        else:
            container2c.bar_chart(data=metrics, stack=False, horizontal=True, height=300)
            # container2c.pyplot(cbr_obj.visualize_metrics(metrics=metrics, col_label=selected_similarity_metrics))

    else:
        st.error("Please upload the test file (csv)")
