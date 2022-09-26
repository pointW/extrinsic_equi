from rl_manipulation.storage.per_buffer import PrioritizedQLearningBuffer
from rl_manipulation.utils.torch_utils import ExpertTransition, augmentTransition
from rl_manipulation.utils.parameters import buffer_aug_type

class PrioritizedQLearningBufferAug(PrioritizedQLearningBuffer):
    def __init__(self, size, alpha, base_buffer, aug_n=9):
        super().__init__(size, alpha, base_buffer)
        self.aug_n = aug_n

    def add(self, transition: ExpertTransition):
        super().add(transition)
        for _ in range(self.aug_n):
            super().add(augmentTransition(transition, buffer_aug_type))





