import torch


class RelLpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(RelLpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.sum(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        target = torch.sum(abs(target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        rloss = torch.mean(error / target)
        return rloss
    

class LpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        loss = torch.mean(error)
        return loss


class RelLpLossWithRecon(torch.nn.modules.loss._Loss):
    """Added 2026-09-23 for the Darcy AE-reconstruction-loss retrofit (see
    claude/convcnp-lno-integration-plan-part4.md) -- ports the
    train_burgers_pdebench.py / train_ns2d_evolution.py --recon_loss_weight
    mechanism (encode+decode round-trip, no propagate() call, as an
    auxiliary loss term) into this shared LNO/exp.py training harness,
    WITHOUT changing exp.py's actual loss-call structure: exp.py still
    just does `loss = loss_fn(res, y2)`, the exact same line as every
    other config.

    The trick: the auxiliary recon term can't be computed from (pred,
    target) alone (it needs the model's own internal encode/decode calls
    on the input and target fields, not just the final prediction), so
    `model` (the DDP-wrapped ConvCNP_LTO) is captured by reference at
    construction time (module/utils.py::get_model_data, right after the
    model is built) rather than passed through forward()'s signature.
    ConvCNP_LTO.forward() computes and stores the (unweighted, averaged)
    recon loss as `self.last_recon_loss` -- a real, still-attached-to-
    the-autograd-graph tensor, not a detached float -- whenever it's
    called in training mode with compute_recon_loss=True; `.module`
    reaches the underlying ConvCNP_LTO through the DDP wrapper. Because
    last_recon_loss is produced by the SAME forward() call that produced
    `pred`, adding it in here and calling .backward() on the sum still
    correctly backpropagates through both the forecast and reconstruction
    paths in one graph -- no second forward pass, no extra DDP
    synchronization needed.

    At eval time (model.eval()), ConvCNP_LTO.forward() always sets
    last_recon_loss to a zero tensor (see its own docstring), so val_loss
    computed through this class is IDENTICAL to plain RelLpLoss -- pure
    forecast rL2, uncontaminated by the auxiliary term, exactly matching
    how val_loss is reported for every other (non-recon) checkpoint in
    this project, so the numbers stay comparable across configs.

    Opt-in only (config.loss.name == "rL2_ae"): every other config keeps
    using plain "rL2"/"L2"/"L1"/"rL1", completely unaffected."""

    def __init__(self, p, model, recon_loss_weight=1.0):
        super(RelLpLossWithRecon, self).__init__()
        self.base = RelLpLoss(p)
        self.model = model
        self.recon_loss_weight = recon_loss_weight

    def forward(self, pred, target):
        base_loss = self.base(pred, target)
        recon_loss = self.model.module.last_recon_loss
        return base_loss + self.recon_loss_weight * recon_loss
