
# MIT License

# Copyright (C) 2023 Dominik Madej

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, ABCMeta, abstractmethod
from ..utils import StrClassNameMeta
from .. import stat


class ParametersEstimatorMeta(StrClassNameMeta, ABCMeta):
    pass


class ParametersEstimator(ABC, metaclass=ParametersEstimatorMeta):

    def __init__(self, scores, hit_index) -> None:
        self.scores = scores
        self.hit_index = hit_index

    @abstractmethod
    def estimate(self):
        pass


class MethodOfMoments(ParametersEstimator):

    def __init__(self, scores, hit_index) -> None:
        super().__init__(scores, hit_index)

    def estimate(self):
        return stat.MethodOfMoments().estimate_parameters(self.scores, self.hit_index)
    

class AsymptoticGumbelMLE(ParametersEstimator):

    def __init__(self, scores, hit_index) -> None:
        super().__init__(scores, hit_index)

    def estimate(self):
        return stat.AsymptoticGumbelMLE(self.scores, self.hit_index).run_mle()


class FiniteNGUmbelMLE(ParametersEstimator):

    def __init__(self, scores, hit_index, num_candidates=1000) -> None:
        super().__init__(scores, hit_index)
        self.num_candidates = num_candidates

    def estimate(self):
        return stat.FiniteNGumbelMLE(self.scores, self.hit_index, self.num_candidates)