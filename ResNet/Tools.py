import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, arch, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.report = None
        self.arch = arch
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model, report):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.report = report
            ExportPATH = f'./Models/{self.arch}'
            torch.save(model.state_dict(), ExportPATH)

            ExportPATH = f'./Models/best'
            torch.save(self.best_score, ExportPATH)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Best curent loss is: {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.report = report
            self.counter = 0
            ExportPATH = f'./Models/{self.arch}'
            torch.save(model.state_dict(), ExportPATH)

            ExportPATH = f'./Models/best'
            torch.save(self.best_score, ExportPATH)