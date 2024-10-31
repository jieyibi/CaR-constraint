import torch
from utils import *

sol = torch.tensor([[[1,2,3,4], [1,2,3,5], [1,2,4,5]]])

calculate_diversity(sol, diversity="jaccard_distance")