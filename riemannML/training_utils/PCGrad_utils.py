import torch 
import torch.nn as nn 
import random 

class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        self._optim.step()

    def pc_backward(self, losses):
        """
        Args:
            losses: list of scalar losses, one per task
        """
        if not isinstance(losses, list):
            raise ValueError("pc_backward expects a list of scalar losses.")

        # Collect per-task gradients
        grads = []
        for loss in losses:
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            single_grads = []
            for param in self._optim.param_groups[0]['params']:
                if param.grad is not None:
                    single_grads.append(param.grad.detach().clone())
                else:
                    single_grads.append(None)
            grads.append(single_grads)

        # Apply projection
        gradsPC = self.apply_PCGrad(losses, grads)

        # Zero gradients again
        self._optim.zero_grad()

        # Overwrite gradients with projected ones
        for k, param in enumerate(self._optim.param_groups[0]['params']):
            if gradsPC[0][k] is None:
                continue
            param.grad = torch.zeros_like(param)
            for task_grad in gradsPC:
                if task_grad[k] is not None:
                    param.grad += task_grad[k]

    def apply_PCGrad(self, losses, grads):
        gradsPC = [[g.clone() if g is not None else None for g in grad] for grad in grads]
        n_tasks = len(losses)

        for i in range(n_tasks):
            for j in self.random_permutation_without_i(n_tasks, i):
                for k, (gradi_k, gradj_k) in enumerate(zip(gradsPC[i], gradsPC[j])):
                    if gradi_k is not None and gradj_k is not None:
                        dot = torch.dot(gradi_k.flatten(), gradj_k.flatten())
                        if dot < 0:  # Conflict detected
                            gradsPC[i][k] = self.project_away(gradi_k, gradj_k)
        return gradsPC

    def random_permutation_without_i(self, N, i):
        perm = list(range(N))
        perm.remove(i)
        random.shuffle(perm)
        return perm

    def project_away(self, gradi, gradj, eps=1e-10):
        dot_ij = torch.dot(gradi.flatten(), gradj.flatten())
        norm_j_sq = torch.dot(gradj.flatten(), gradj.flatten()) + eps
        return gradi - (dot_ij / norm_j_sq) * gradj


