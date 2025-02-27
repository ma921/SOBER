# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause

from botorch.acquisition.analytic import UpperConfidenceBound
from numpy import log, pi


class SOBERUCB:
    def __init__(self, model, label="UCB", sample_size=1, exploration_rate=10):
        self.label = label
        self.exploration_rate = exploration_rate
        self.beta = (2 * log(
            sample_size**(2 * model.dim + 2) * pi**2
            / (3 / self.exploration_rate)
        ))
        self.af = UpperConfidenceBound(model, beta=self.beta)

    def __call__(self, x):
        return self.af(x.unsqueeze(1)).detach()
