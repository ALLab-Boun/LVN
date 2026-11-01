from copy import deepcopy


class EarlyStopping:
    """Keeps the best model state by the monitored metric. The patience counter is reset
    only when the metric improves by more than 1% relative to the last reset."""

    def __init__(self, patience, start_epoch):
        self.patience = patience
        self.start_epoch = start_epoch
        self.best = 0.0
        self.goal = 0.0
        self.count = 0
        self.best_state = None

    def step(self, metric, model, epoch, eval_freq=1):
        if metric > self.goal:
            self.best, self.count, self.goal = metric, 0, metric * 1.01
            self.best_state = deepcopy(model.state_dict())
        else:
            if metric > self.best:
                self.best = metric
                self.best_state = deepcopy(model.state_dict())
            self.count += eval_freq
        return self.count >= self.patience and epoch >= self.start_epoch
