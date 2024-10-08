class ExptSettings:
    def __init__(self):
        self.label = "base"
        self.seed = 0

        # Data
        self.p = 97
        self.task = "multiplication"
        self.split_ratio = 0.5
        
        self.budget = 3e5
        self.batch_size = 512
        self.lr = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.weight_decay = 1.0
        self.optimizer = "AdamW"
        
        # Grokfast
        self.filter = "none"  # choices: ["none", "ma", "ema", "fir"]
        self.alpha = 0.99
        self.window_size = 100
        self.lamb = 5.0

        # Ablation studies
        self.two_stage = False
        self.save_weights = False

        # Hessian
        self.hessian_save_every = 20
        self.explicit_hessian_regularization = 0.0

        # NSM
        self.nsm = False
        self.nsm_sigma = 0.01
        self.nsm_distribution = "normal"