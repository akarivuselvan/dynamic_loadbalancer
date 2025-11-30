# Baseline Strategies: Round Robin and Random

import random


class RoundRobinStrategy:
    def __init__(self, num_vms):
        self.num_vms = num_vms
        self.counter = 0
        self.name = "RoundRobin"

    def assign_batch(self, tasks, vms, current_time):
        for task in tasks:
            vm_idx = self.counter % self.num_vms
            self.counter += 1

            task.assigned_vm = vm_idx
            task.start = max(task.arrival, current_time)
            vms[vm_idx].queue.append(task)


class RandomStrategy:
    def __init__(self, num_vms, seed=None):
        self.num_vms = num_vms
        self.name = "Random"
        if seed is not None:
            random.seed(seed)

    def assign_batch(self, tasks, vms, current_time):
        for task in tasks:
            vm_idx = random.randrange(self.num_vms)
            task.assigned_vm = vm_idx
            task.start = max(task.arrival, current_time)
            vms[vm_idx].queue.append(task)
