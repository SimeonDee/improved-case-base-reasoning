# An Improved Case Retrieval Model for Case-Base Reasoning Systems with a Hybridized Geometric Similarity Measure

An Improved Case-Based Reasoning (CBR) System implemented as an hybridization of (Euclidean distance, cosine and jaccard similarity scores). 

It uses streamlit to render a friendly user interface for end-users.

Users select from a list of symptoms, sets a number of desired k-nearest solutions and a similarity threshold, then the system uses CBR approach to return the k-disease (Nearest solution) having similarity scores above the specified threshold.

---
- Name: `Oluwaseyi Oluwatobi OMONIJO`
- Matric: `LCU/PG/001401`

---

## Setup
- Install dependencies: 
    - Open Terminal (MAC) or  Command Prompt (Windows) or Shell (Linux) and run the following command
    - Navigate to the working directory
    - Create a virtual environment (Optional). See why and how to [create virtual environment here](https://www.geeksforgeeks.org/python-virtual-environment/)
    - run the command below to install the dependencies:

        ```bash
        $ pip install -r requirements.txt
        ```

## Runing the application
```bash
(env)$ streamlit run ./Main_Page.py
````

## Stop running server:
- To stop the running server, go to the command prompt or terminal and press `Cntrl + C` button

NOTE:
See [documentation here](./DOC.md)