import numpy as np
import pandas as pd


class Classifier:
    def __init__(self, k_min=10, gamma=0):
        ''' k_min: min. batch size
                  gamma: time difference within batch processing'''
        self.k_min = k_min
        self.gamma = gamma
        self.observations = pd.DataFrame()

    def batch_labeler(self, temp_batch, batch_num):
        self.observations.loc[
            (self.observations.index >= temp_batch[0]) & (
                    self.observations.index <= temp_batch[-1]), 'class'] = batch_num

    def batch_looper(self, to_do, batching_condition, batch_num=1):
        temp_batch = []
        temp_batch.append(0)
        for j in to_do:
            if batching_condition(j):
                temp_batch.append(j)
                if j == max(to_do) - 1 and len(temp_batch) >= self.k_min:
                    self.batch_labeler(temp_batch, batch_num)
                    batch_num += 1
            else:
                if len(temp_batch) >= self.k_min:
                    self.batch_labeler(temp_batch, batch_num)
                    batch_num += 1
                temp_batch = []
                temp_batch.append(j)
        return batch_num

    def end_batching_condition(self, j):
        return ((self.observations['end_time'][j - 1] <= self.observations['end_time'][j] <= self.gamma +
                 self.observations['end_time'][
                     j - 1]) and (self.observations['start_time'][j] >= self.observations['start_time'][j - 1]))

    def start_batching_condition(self, j):
        return ((self.observations['class'][j] == 0) and j < len(self.observations) - 1) and (
                (self.observations['start_time'][j - 1] <= self.observations['start_time'][j] <= self.gamma +
                 self.observations['start_time'][j - 1]) and (
                        self.observations['end_time'][j + 1] > self.observations['end_time'][j] >=
                        self.observations['end_time'][
                            j - 1]))

    def classify_batch(self, df):
        '''Input: df containing the columns ['start_time', 'end_time', 'resource']

           Output: Batch Classification'''

        # Setting up the df:
        df_sorted = df.sort_values(by=['start_time', 'end_time'], axis=0)
        if len(df_sorted) == 0:  # If the DataFrame is empty, don't run.
            return []
        self.observations = df_sorted.reset_index().copy()
        self.observations['class'] = np.zeros(len(self.observations))

        # End batching:
        print("start end-batching")
        to_do = range(1, len(self.observations))
        max_batch_num = self.batch_looper(to_do, self.end_batching_condition)

        # Start batching:
        print("start start-batching")
        to_do = range(1, len(self.observations))
        self.batch_looper(to_do, self.start_batching_condition, batch_num=max_batch_num)
        return list(self.observations.sort_values(by='index')['class'])

    # def classify_batch_by_resource(self):
    #     batches = self.observations[self.observations['class'] != 0]
    #     resources = batches['resource'].unique()
    #     for resource in resources:
