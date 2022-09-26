from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function
)

import SyntheticMultiLoggersBanditDataset

# generate synthetic contextual bandit feedback with 10 actions.
'''dataset = SyntheticBanditDataset(
    n_actions=3,
    dim_context=5,
    reward_function=logistic_reward_function,
    random_state=12345
)
bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
print(bandit_feedback)'''

# generate synthetic contextual bandit feedback with 10 actions from new class.
dataset2 = SyntheticMultiLoggersBanditDataset.SyntheticMultiLoggersBanditDataset(
    n_actions=10,
    dim_context=5,
    reward_function=logistic_reward_function,
    betas=[-3, 0, 3],
    rhos=[0.2, 0.5, 0.3],
    random_state=12345
)
bandit_feedback2 = dataset2.obtain_batch_bandit_feedback(n_rounds=100000)
print(bandit_feedback2)
