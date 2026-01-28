import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentTradingLoss(nn.Module):
    def __init__(self, tokenizer_centers, risk_free_rate=0.0, focal_gamma=2.0):
        """
        model_logits:
        tokenizer_centers: Tensor of bin center values (e.g., predicted prices)
        """
        super().__init__()
        self.centers = tokenizer_centers
        self.rfr = risk_free_rate
        self.gamma = focal_gamma

        # Learnable weights for Homoscedastic Uncertainty (3 tasks)
        # We initialize with 0.0 (which equates to sigma=1.0)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, student_logits, teacher_logits, targets_tbm):
        """
        student_logits: Output from Chronos Bolt (Student)
        teacher_logits: Output from Chronos T5 (Teacher)
        targets_tbm: Triple Barrier Method Labels (1 = Buy, 0 = Neutral/Sell)
        """

        # --- TASK 1: Distillation (KL Divergence) ---
        # "Learn the Wisdom"
        # We use log_softmax for student and softmax for teacher (standard KL formulation)
        # KL(P || Q) = sum P(x) log(P(x)/Q(x)) ?
        # PyTorch KLDivLoss expects input=log_prob, target=prob (if reduction='batchmean')
        # But here input is Student (Q?), target is Teacher (P?).
        # Usually Distillation: Minimize KL(Teacher || Student) or KL(Student || Teacher)?
        # Hinton: KL(Teacher || Student) -> Student should match Teacher.
        # Torch `kl_div(input, target)` computes sum(target * (log(target) - input)).
        # If input is log_softmax(Student), target is softmax(Teacher).
        # Then it minimizes distance.

        loss_distill = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )

        # --- TASK 2: Differentiable Sortino Ratio ---
        # "Learn to Earn"

        # 1. Convert logits to probabilities
        probs = F.softmax(student_logits, dim=-1) #

        # 2. Calculate Expected Price (Differentiable)
        # centers needs shape to broadcast
        # centers is likely (num_bins) or (1, 1, num_bins)
        if self.centers.device != probs.device:
            self.centers = self.centers.to(probs.device)

        centers_broadcast = self.centers.view(1, 1, -1)
        expected_prices = torch.sum(probs * centers_broadcast, dim=-1) #

        # 3. Calculate Returns from Expected Prices
        # We assume sequence length > 1
        returns = torch.diff(expected_prices, dim=1) / expected_prices[:, :-1]

        # 4. Sortino Calculation
        # We only penalize downside deviation (returns < rfr)
        downside_returns = torch.clamp(returns - self.rfr, max=0)

        # Use simple squared mean for downside deviation (keeping it stable)
        downside_std = torch.sqrt(torch.mean(downside_returns**2, dim=1) + 1e-6)
        mean_return = torch.mean(returns, dim=1)

        sortino_ratio = mean_return / (downside_std + 1e-6)

        # We want to MAXIMIZE Sortino, so we MINIMIZE negative Sortino
        loss_sortino = -torch.mean(sortino_ratio)

        # --- TASK 3: Focal Loss on "Implied" Classification ---
        # "Focus on Opportunities"

        # Instead of a separate head, we calculate the probability mass
        # in the "upper bins" (e.g., bins representing >0.5% return)
        # Assume upper_half_indices are indices of bins > threshold
        # For simplicity here, we sum the top 20% of bins as "Buy Probability"
        num_bins = student_logits.shape[-1]
        top_bins = int(num_bins * 0.2)

        # Sum prob of top bins to get p(Buy)
        buy_prob = torch.sum(probs[:, :, -top_bins:], dim=-1) #
        buy_prob = torch.mean(buy_prob, dim=1) # Average prob over sequence

        # Standard Focal Loss formula
        # Loss = - alpha * (1-pt)^gamma * log(pt)
        # Here we just use the simple binary version without alpha for now
        # Targets need to be same shape as buy_prob (Batch,)
        # targets_tbm might be (Batch, Time) or (Batch,).
        # If (Batch, Time), we need to aggregate or use sequence.
        # The prompt code assumes `targets_tbm.float()`.
        # `buy_prob` is (Batch,). So targets_tbm should be (Batch,) or we average?
        # In distillation loop, targets_tbm is likely passed as (Batch,) representing the label for the window?
        # Or (Batch, L)?
        # The code `buy_prob = torch.mean(buy_prob, dim=1)` reduces time dim.
        # So it classifies the whole window?
        # If targets_tbm is (Batch, L), we should take mean or last?
        # For now, let's assume targets_tbm is (Batch,) or we reduce it.
        # If targets_tbm is (Batch, L):
        if targets_tbm.ndim > 1:
            targets = targets_tbm.float().mean(dim=1) # Soft label if multiple?
            # Or assume targets_tbm is constant for the window?
            # Usually we predict "Is this window a buy?"
            # Let's assume (Batch, L) and we check if ANY is 1? Or mean?
            # Code uses `targets = targets_tbm.float()`.
            # If shape mismatch, BCE will complain.
            pass
        else:
            targets = targets_tbm.float()

        # Check shape
        if targets.shape != buy_prob.shape:
             # Try to adapt
             if targets.numel() == buy_prob.numel():
                 targets = targets.view_as(buy_prob)
             else:
                 # If targets is (B, L) and buy_prob is (B,), maybe we want loss per step?
                 # But sortino was per sequence.
                 # Let's keep buy_prob aggregated.
                 # If targets is (B, L), max?
                 targets, _ = torch.max(targets, dim=1) # If any 1, target is 1

        bce_loss = F.binary_cross_entropy(buy_prob, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss_focal = ((1 - pt) ** self.gamma * bce_loss).mean()

        # --- COMBINE WITH HOMOSCEDASTIC WEIGHTING ---
        # Loss = 1/(2*sigma^2) * Loss_i + log(sigma)

        # Weight 1: Distillation
        precision1 = torch.exp(-self.log_vars[0])
        l1 = precision1 * loss_distill + self.log_vars[0]

        # Weight 2: Sortino
        precision2 = torch.exp(-self.log_vars[1])
        l2 = precision2 * loss_sortino + self.log_vars[1]

        # Weight 3: Focal
        precision3 = torch.exp(-self.log_vars[2])
        l3 = precision3 * loss_focal + self.log_vars[2]

        total_loss = l1 + l2 + l3

        return total_loss, (l1.item(), l2.item(), l3.item())
