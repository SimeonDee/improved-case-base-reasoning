import numpy as np
from scipy import spatial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ImprovedCBR:
    def __init__(self, knowldge_base_df: pd.DataFrame):
        if not isinstance(knowldge_base_df, pd.DataFrame):
            raise Exception("'knowldge_base_df' has to be a pandas DataFrame object")
        
        self.kb = knowldge_base_df
        self.current_problem = None
        self.current_solutions = None
    

    ####################################
    ## CALCULATES SIMILARITY SCORES
    ####################################
    def binary_euclidean_distance(self, vector1, vector2):
        """
        Calculates Euclidean distance between two binary vectors

        Args:
        vector1 (list or numpy array): First binary vector
        vector2 (list or numpy array): Second binary vector

        Returns:
        float: Euclidean distance between vectors in range(0: "identical", sqrt(n):"not-identical")
        """

        return np.linalg.norm(np.array(vector1) - np.array(vector2))


    def binary_cosine_similarity(self, vector1, vector2):
        """
        Calculates cosine similarity between two binary vectors

        Args:
        vector1 (list or numpy array): First binary vetcor
        vector2 (list or numpy array): Second binary vetcor

        Returns:
        float: Cosine similarity between vectors in range(-1:"not-identical", 1:"identical")
        """

        return 1 - spatial.distance.cosine(vector1, vector2)


    def binary_jaccard_similarity(self, vector1, vector2):
        """
        Calculates jaccard similarity between two binary vectors

        Args:
        vector1 (list or numpy array): First binary vetcor
        vector2 (list or numpy array): Second binary vetcor

        Returns:
        float: Jaccard similarity between vectors, value in range(0:"not-identical", 1:"identical")
        """

        intersection = np.sum(np.logical_and(vector1, vector2))
        union = np.sum(np.logical_or(vector1, vector2))

        return intersection / union


    def compute_distance_scores(self, problem, base_cases_df, solutions_df):
        """
        Calculates the euclidean, cosine and jaccard similarity scores between 'problem' and 'base_cases_df'

        Args:
        problem (list): problem case whose solution we wish to find from the case-base
        base_cases_df (DataFrame Object): All exising problem cases in the knowledge base
        solutions_df (DataFrame Object): All corresponding solutions for the retrieved existing base_cases_df

        Returns: 
        DataFrame: computed distance scores and the attached solutions, for further processing
        """
        euclid_scores = []
        cosine_scores = []
        jaccard_scores = []

        for i in range(len(base_cases_df)):
            current_case = list(base_cases_df.iloc[i].values)
            
            euclid_score = self.binary_euclidean_distance(problem, current_case)
            cosine_score = self.binary_cosine_similarity(problem, current_case)
            jaccard_score = self.binary_jaccard_similarity(problem, current_case)

            euclid_scores.append(np.round(euclid_score, 2))
            cosine_scores.append(np.round(cosine_score, 2))
            jaccard_scores.append(np.round(jaccard_score, 2))
        
        results = solutions_df.copy()
        results['euclid_scores'] = euclid_scores
        results['cosine_scores'] = cosine_scores
        results['jaccard_scores'] = jaccard_scores

        return results


    ##############################################
    # Scaling and Standardizing the scores
    ##############################################

    def scale_euclid_cosine_jaccard_score(self, score, metric='euclidean'):
        """
        Scale 'score' to range [0,1] from 
        - values in range [0,6] for euclidean distance using min-max scaling, 
        - value in range [-1,1] for cosine similary, using linear transformation and
        - value in range [0,1] for jaccard similarity using min-max scaling

        Args:
        score (int): the score to be scaled
        metric (string): metric indicating the scale to convert. 
            Values: {'euclidean', 'cosine', 'jaccard'}, defaults to 'euclidean'

        Returns:
        float: a scaled score in range [0,1]
        """

        scaled_result = None

        if metric == 'euclidean':
            # min_value (int): minimum possible value of 'score' (Identical). Defaults to 0 
            # max_value (int): maximum possible value of 'score' (Not-Identical). Defaults to 6
            # NOTE: The lesser the distance 'score', the more the similarity.

            min_value=0
            max_value=6    

            scaled_score = ((score - min_value) / (max_value - min_value)) 
            scaled_result = 1 - scaled_score # inverted score (1 - scaled score)

        elif metric == 'cosine':
            # Scale a cosine score using simple linear transformation from the range [-1,1] to range [0,1].

            # min_value=-1
            # max_value=1
            # min_value (int): minimum possible value of 'score' (Not-identical). Defaults to -1 
            # max_value (int): maximum possible value of 'score' (Identical). Defaults to 1
            # NOTE: The more the similarity 'score', the better the similarity.
            
            scaled_result =  (score + 1) / 2
        
        elif metric == 'jaccard':
            # Scale a jaccard score using min-max scaler formula from the range [0,1] to range [0,1].
        
            min_value=0
            max_value=1
            # min_value (int): minimum possible value of 'score' (Not-identical). Defaults to 0
            # max_value (int): maximum possible value of 'score' (Identical). Defaults to 1
            
            scaled_result =  (score - min_value) / (max_value - min_value)

        else:
            raise ValueError("Wrong value passed for parameter 'metric'. Allowed values are {'euclidean', 'cosine', 'jaccard'}")
            scaled_result = None
        
        return np.round(scaled_result, 2)


    def generic_scaler(self, value:int, min_value:int, max_value:int, new_min:int, new_max:int, metric:str='cosine'):
        """
        Scales a given 'value' in range [min_value, max_value] to a value in range [new_min, new_max] using 'metric' metric

        Args:
        min_value (int): minimum possible value of 'score' (Not-identical). Defaults to 0
        max_value (int): maximum possible value of 'score' (Identical). Defaults to 1
        new_min (int): minimum possible value of the output desired (Not-identical).
        new_max (int): maximum possible value of the output desired (identical).
        metrics (string): {'cosine', 'euclidean', 'jaccard'}

        Return:
        float: scaled value in range [new_min, new_max]
        """
        scaled_score = ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
        if metric == 'eucliean':
            return 1 - scaled_score
        else:
            return scaled_score

    def get_best_k_cases(self, cases_results, k=1, thresh_hold=0.70):
        answers = cases_results.sort_values(by='weighted_scores', ascending=False)

        answers = answers[answers['weighted_scores'] >= thresh_hold]
        
        return answers[:k] if len(answers) >= k else answers

    def solve_problem(self, problem:list, k:int=1, thresh_hold:float=0.75):
        """
        Finds the 'k' solutions in the Knowledge-base whose cases have 
        similarity score above the specified 'thresh_hold' with the given 'problem'
        """
        base_cases = self.kb[self.kb.columns[:-2]]
        solutions = self.kb[self.kb.columns[-2:]]

        # problem = list(base_cases.iloc[0].values)

        # calculating the distance scores
        print("Computing Similarity measures (Euclidean, Cosine and Jaccard similarity measures)...")
        distance_scores = self.compute_distance_scores(problem=problem, base_cases_df=base_cases, solutions_df=solutions)
        print("Computing Similarity measures... Done.")
        
        print("Scaling/Standardizing computed similarity measures...")
        # Scaling all distance scores to values in range(0,1) for standardization of scores
        # Euclidean distance: 
            # min=0 (identical), max=6 (not-identical) 
            # Note: {max = sqrt(n), where n is the total number of all symptoms (33). i.e. 5.7 = 6 (approx.)}
        # Cosine similarity: 
            # min=-1 (not-identical), max=1 (identical)
        # Jaccard similarity: 
            # min=0 (not-identical), max=1 (identical)

        print("\t Scaling euclidean distance scores ...")
        distance_scores['scaled_euclid_scores'] = distance_scores.euclid_scores.apply(lambda x: self.scale_euclid_cosine_jaccard_score(x, metric='euclidean'))
        print("\t Scaling euclidean distance scores ... Completed")
        
        print("\t Scaling Cosine-Similarity scores ...")
        distance_scores['scaled_cosine_scores'] = distance_scores.cosine_scores.apply(lambda x: self.scale_euclid_cosine_jaccard_score(x, metric='cosine'))
        print("\t Scaling Cosine-Similarity scores ... Completed")

        print("\t Scaling Jaccard-Similarity scores ...")
        distance_scores['scaled_jaccard_scores'] = distance_scores.jaccard_scores.apply(lambda x: self.scale_euclid_cosine_jaccard_score(x, metric='jaccard'))
        print("\t Scaling Jaccard-Similarity scores ... Completed")

        print("Scaling/Standardizing computed similarity measures... Done.")


        ## Compute Weighted Scores
        print("Calculating weighted scores...")

        distance_scores['weighted_scores'] = distance_scores[[
            'scaled_euclid_scores', 
            'scaled_cosine_scores',
            'scaled_jaccard_scores'
        ]].mean(axis=1).round(2)

        print("Calculating weighted scores... Done.")


        ## Get best k solutions above given thresh_hold
        ###################################################
        
        print(f"Retrieving best {k} Solution(s) from Knowledge base, above {thresh_hold} thresh-hold...")
        final_results_df = self.get_best_k_cases(distance_scores, k=k, thresh_hold=thresh_hold)
        print(f"Found best {len(final_results_df)} Solution(s) from Knowledge base, above {thresh_hold} thresh-hold... Done")

        self.current_problem = problem
        self.current_solution = final_results_df

        return final_results_df
    
    def save_new_row(self, row:list=None, filename='Dataset/kb_updated.csv'):
        col_names = self.kb.columns

        if len(row) != len(col_names):
            raise Exception(f'"row" must be a list of size {len(col_names)}')
        
        next_index = self.get_next_index()
        new_row = np.array(row).reshape(-1, len(col_names))
        new_row = pd.DataFrame(new_row, columns=col_names, index=[next_index])

        # add new row to KB and persist to disk
        self.kb = pd.concat([self.kb, new_row], axis='index')
        self.persist_kb(filename=filename)


    def get_next_index(self):
        split_last_index = self.kb.index[-1].split('_')
        new_index = f"{split_last_index[0]}_{str(int(split_last_index[1]) + 1).rjust(3,'0')}"
        return new_index
    
    def persist_kb(self, filename:str=''):
        if filename == '':
            filename = 'Dataset/COLIBRI Training cbr_case_dataset Boolean USING.csv'
        
        # Persist to disk
        self.kb.to_csv(filename)

    def visualize_results(self, results:pd.DataFrame):
        fig = plt.figure(figsize=(8,3))
        # plt.barh(data=final_results.sort_values(by='weighted_scores'), 
        #                  y='CaseAlias', width='weighted_scores', align='center')
        sns.barplot(data=results.sort_values(by='weighted_scores', ascending=False), 
                        y='CaseAlias', x='weighted_scores', hue='CaseAlias')
        plt.title('Solutions\n-------------\n')
        plt.ylabel('Predicted Diseases')
        plt.xlabel('Weighted Similarity Score')

        return fig