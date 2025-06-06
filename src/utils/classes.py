'''CLASSES MODULE'''
import numpy as np

class CustomDistribution:
    '''
    Class to model categorical variable distribution
    '''

    def __init__(self, buckets, percentages):
        self.buckets = buckets
        self.percentages = percentages
        self.cumulative_percentages = np.cumsum(percentages)

    def sample(self, size=1):
        rand_nums = np.random.rand(size)
        indices = np.searchsorted(self.cumulative_percentages, rand_nums)
        samples = []
        for i in indices:
            low, high = self.buckets[i]
            samples.append(np.random.uniform(low, high))
        return np.array(samples)
