## Help on class ImprovedCBR in module utils:

class ImprovedCBR(builtins.object)

---

 |  **ImprovedCBR(knowldge_base_df: pandas.core.frame.DataFrame)**
 |  
 |  Methods defined here:
 |  
 |  __init__(self, knowldge_base_df: pandas.core.frame.DataFrame)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  **binary_cosine_similarity(self, vector1, vector2)**
 |      Calculates cosine similarity between two binary vectors
 |      
 |      Args:
 |      vector1 (list or numpy array): First binary vetcor
 |      vector2 (list or numpy array): Second binary vetcor
 |      
 |      Returns:
 |      float: Cosine similarity between vectors in range(-1:"not-identical", 1:"identical")
 |  
 |  **binary_euclidean_distance(self, vector1, vector2)**
 |      Calculates Euclidean distance between two binary vectors
 |      
 |      Args:
 |      vector1 (list or numpy array): First binary vetcor
 |      vector2 (list or numpy array): Second binary vetcor
 |      
 |      Returns:
 |      float: Euclidean distance between vectors in range(0: "identical", sqrt(n):"not-identical")
 |  
 |  **binary_jaccard_similarity(self, vector1, vector2)**
 |      Calculates jaccard similarity between two binary vectors
 |      
 |      Args:
 |      vector1 (list or numpy array): First binary vetcor
 |      vector2 (list or numpy array): Second binary vetcor
 |      
 |      Returns:
 |      float: Jaccard similarity between vectors, value in range(0:"not-identical", 1:"identical")
 |  
 |  **compute_distance_scores(self, problem, base_cases_df, solutions_df)**
 |      Calculates the euclidean, cosine and jaccard similarity scores between 'problem' and 'base_cases_df'
 |      
 |      Args:
 |      problem (list): problem case whose solution we wish to find from the case-base
 |      base_cases_df (DataFrame Object): All exising problem cases in the knowledge base
 |      solutions_df (DataFrame Object): All corresponding solutions for the retrieved 
 |               existing base_cases_df
 |      
 |      Returns: 
 |      DataFrame: computed distance scores and the attached solutions, for further processing
 |  
 |  **generic_scaler(self, value: int, min_value: int, max_value: int, new_min: int, new_max: int, metric: str = 'cosine')**
 |      Scales a given 'value' in range [min_value, max_value] to a value in range [new_min, new_max] using 'metric' metric
 |      
 |      Args:
 |      min_value (int): minimum possible value of 'score' (Not-identical). Defaults to 0
 |      max_value (int): maximum possible value of 'score' (Identical). Defaults to 1
 |      new_min (int): minimum possible value of the output desired (Not-identical).
 |      new_max (int): maximum possible value of the output desired (identical).
 |      metrics (string): {'cosine', 'euclidean', 'jaccard'}
 |      
 |      Return:
 |      float: scaled value in range [new_min, new_max]
 |  
 |  **get_best_k_cases(self, cases_results, k=1, thresh_hold=0.7)**
 |  
 |  **get_next_index(self)**
 |  
 |  **persist_kb(self, filename: str = '')**
 |  
 |  **save_new_row(self, row: list = None, filename='Dataset/kb_updated.csv')**
 |  
 |  **scale_euclid_cosine_jaccard_score(self, score, metric='euclidean')**
 |      Scale 'score' to range [0,1] from 
 |      - values in range [0,6] for euclidean distance using min-max scaling, 
 |      - value in range [-1,1] for cosine similary, using linear transformation and
 |      - value in range [0,1] for jaccard similarity using min-max scaling
 |      
 |      Args:
 |      score (int): the score to be scaled
 |      metric (string): metric indicating the scale to convert. 
 |          Values: {'euclidean', 'cosine', 'jaccard'}, defaults to 'euclidean'
 |      
 |      Returns:
 |      float: a scaled score
 |  
 |  **solve_problem(self, problem: list, k: int = 1, thresh_hold: float = 0.75)**
 |      Finds the 'k' solutions in the Knowledge-base whose cases have 
 |      similarity score above the specified 'thresh_hold' with the given 'problem'
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables
 |  
 |  __weakref__
 |      list of weak references to the object
