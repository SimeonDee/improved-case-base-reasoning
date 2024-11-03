from utils import ImprovedCBR
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/COLIBRI Training cbr_case_dataset Boolean USING.csv", index_col=0)

# instantiating the cbr class
cbr = ImprovedCBR(df)
problem = []

# set page to wide screen by default
st.set_page_config(layout="wide")

st.write("""
         <h2 style="text-align:center;">An Improved Case Retrieval Model for Case-Base Reasoning Systems with a Hybridized Geometric Similarity Measure</h2>
         
         ---
         <h4 style="text-align:center;">Oluwaseyi Oluwatobi OMONIJO</h4>
         <h4 style="text-align:center;">(LCU/PG/001401) </h4>

         ---
         """, unsafe_allow_html=True)

ct = st.container(border=True)

ct.write("#### CBR for Cat Fish Disease Diagnosis")

ct1 = ct.container(border=True)

ct1.write("<b>Kindly Answer the following questions by checking the Symptoms the Fish has</b>", unsafe_allow_html=True)

col1,col2,col3 = ct1.columns([2,2,2])
s1 = col1.toggle('Lethargic Behaviour', value=False)
s2 = col1.toggle('Exophthalmia (Pop eyes: either or both eyes)', value=False)
s3 = col1.toggle('Loss of appetite', value=False)
s4 = col1.toggle('Erratic/abnormal swimming', value=False)
s5 = col1.toggle('Rapid gill movement', value=False)
s6 = col1.toggle('Breathing difficulty', value=False)
s7 = col1.toggle('Frayed/eroded gil filament', value=False)
s8 = col1.toggle('Skin ulcers/lesions', value=False)
s9 = col1.toggle('Cotton-like growth on affected areas', value=False)
s10 = col1.toggle('Camped fins', value=False)
s11 = col1.toggle('Excess mucus production on gills/skin', value=False)

s12 = col2.toggle('Ragged/frayed fins and/or tails', value=False)
s13 = col2.toggle('Scratching body agst hard object (flashing)', value=False)
s14 = col2.toggle('Swollen abdomen (ascites)', value=False)
s15 = col2.toggle('Visible flukes under migration', value=False)
s16 = col2.toggle('Discoluoration of skin', value=False)
s17 = col2.toggle('Pale gills', value=False)
s18 = col2.toggle('Sunken eyes (endophthalmia)', value=False)
s19 = col2.toggle('Fluid accumulation under the skin', value=False)
s20 = col2.toggle('Curved spine', value=False)
s21 = col2.toggle('Cloloured spot/nodles on skin/gill/fin', value=False)
s22 = col2.toggle('Swollen anus', value=False)

s23 = col3.toggle('Gasping at the water surface', value=False)
s24 = col3.toggle('Hemorrhagic margins/signs', value=False)
s25 = col3.toggle('Abscesses (internal and external)', value=False)
s26 = col3.toggle('Whitish eyes (corneal opacity)', value=False)
s27 = col3.toggle('Muscle twitching', value=False)
s28 = col3.toggle('Loss of vision (retinal lesion)', value=False)
s29 = col3.toggle('Emaciation', value=False)
s30 = col3.toggle('Visible parasite on the skin under magnification', value=False)
s31 = col3.toggle('Long, thin worms protruding from the body surface/skin', value=False)
s32 = col3.toggle('Mottled gills', value=False)
s33 = col3.toggle('Swollen gills', value=False)

ct2 = ct.container(border=True)
k = ct2.number_input("Number of solutions to retrieve (k):", value=4)
thresh_hold = ct2.slider("Minimum Similarity thresh_hold (0-1)", min_value=0.0, max_value=1.0, value=0.70, step=0.05)

submit_button = ct.button("Retrieve Case Solutions", type='primary')

if submit_button:
    problem.append(s1)
    problem.append(s2)
    problem.append(s3)
    problem.append(s4)
    problem.append(s5)
    problem.append(s6)
    problem.append(s7)
    problem.append(s8)
    problem.append(s9)
    problem.append(s10)
    problem.append(s11)

    problem.append(s12)
    problem.append(s13)
    problem.append(s14)
    problem.append(s15)
    problem.append(s16)
    problem.append(s17)
    problem.append(s18)
    problem.append(s19)
    problem.append(s20)
    problem.append(s21)
    problem.append(s22)

    problem.append(s23)
    problem.append(s24)
    problem.append(s25)
    problem.append(s26)
    problem.append(s27)
    problem.append(s28)
    problem.append(s29)
    problem.append(s30)
    problem.append(s31)
    problem.append(s32)
    problem.append(s33)

    problem = [int(symptom) for symptom in problem]
    
    if len(problem) < 33:
        size = 33 - len(problem)
        padding_values = list(np.zeros(size, dtype=np.int8))
        
        problem.extend(padding_values)

    solutions = cbr.solve_problem(problem=problem, k=k, thresh_hold=thresh_hold)


    if len(solutions) > 0:
        ct3 = st.container(border=True)
        ct3.write("""
                  <h2 style="text-align:center;"> Results </h2>

                  ---
                  """, unsafe_allow_html=True)
        
        ct3.write(f"#### Found {len(solutions)} solution(s) with similarity score above {thresh_hold} thresh-hold")
        result_ct = ct3.container(border=True)
        result_ct.write("<b>Retrieved Soltions</b>", unsafe_allow_html=True)

        ct4 = ct3.container(border=True)
        ct4.dataframe(solutions)

        predictions = solutions[['CaseAlias', 'weighted_scores']]
        
        ct5 = ct3.container(border=True)
        ct5.write(f"""
                  <h4 style="text-align:center">The {len(predictions)}-Nearest (Best) Solutions</h4>
                  
                  ---
                  """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = ct5.columns([1,2,2,1])
        
        col2.write(predictions)

        # fig = plt.figure(figsize=(5,3))
        # plt.barh(data=predictions.sort_values(by='weighted_scores'), 
        #          y='CaseAlias', width='weighted_scores', align='center')
        # plt.title('Solutions')
        # plt.xlabel('Similarity Scores')

        fig = cbr.visualize_results(predictions)
        col3.pyplot(fig)

        btn_save_best_solution = st.button("Save Best Solution", type='primary')

        if btn_save_best_solution:
            best_solution = predictions.iloc[0]
            best_weight = best_solution['weighted_scores'].values[0]

            solution_values = []
            
            st.write(f'Best Weight: {best_weight}')

            if best_weight > thresh_hold and best_weight < 1:
                solution_values = list(best_solution.values[:2])

                new_row = problem.extend(solution_values)
                print(new_row)

                # cbr.save_new_row(new_row)
                st.success(f"Knowledge-Base updated successfully. current problem case and its best solution {solution_values[1]} saved.")
            
            elif best_weight >= thresh_hold and best_weight == 1:
                st.info('Current problem and its best solution match already exists in knowledge base')
    
    else:
        st.error(f"#### No solution found for the problem-case with similarity score above {thresh_hold} thresh-hold")