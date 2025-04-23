# Losses for CRC-RL: CRC, contrastive, reconstruction, consistency
import torch
import torch.nn.functional as F

def contrastive_loss(q, k, temperature=0.1):
    # q, k: [batch, dim]
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = torch.matmul(q, k.T) / temperature
    labels = torch.arange(q.size(0)).long().to(q.device)
    return F.cross_entropy(logits, labels)

def reconstruction_loss(recon, target):
    return F.mse_loss(recon, target)

def consistency_loss(q1, q2):
    return F.mse_loss(q1, q2)

def crc_loss(q, k, recon, target, q1, q2, alpha=1.0, beta=1.0, gamma=1.0):
    # CRC = alpha*contrastive + beta*reconstruction + gamma*consistency
    l_contrast = contrastive_loss(q, k)
    l_recon = reconstruction_loss(recon, target)
    l_consist = consistency_loss(q1, q2)
    return alpha * l_contrast + beta * l_recon + gamma * l_consist
