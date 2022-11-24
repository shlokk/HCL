import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb as pdb_original
import sys
import math
import moco.hyptorch.nn as hypnn

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, c=1, train_x=False, train_c=False, decoupled=False, theta=1.):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.theta = theta
        self.eps = 1e-5
        self.decoupled = decoupled

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        dim_fc_out = self.encoder_q.fc.weight.shape[0]
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # self.tp = hypnn.ToPoincare(c=c, train_x=train_x, train_c=train_c, ball_dim=dim_fc_out, riemannian=False) # v2.05.04 no RSGD
        self.tp = hypnn.ToPoincare(c=c, train_x=train_x, train_c=train_c, ball_dim=dim_fc_out)
        # self.mlr = hypnn.HyperbolicMLR(ball_dim=dim_mlp, n_classes=K+1, c=c)
        self.hyperbolic_dis = hypnn.HyperbolicDistanceLayer(c=c)

        # create the queue
        self.register_buffer("queue_c", torch.randn(dim, K))
        self.queue_c = nn.functional.normalize(self.queue_c, dim=0)
        self.register_buffer("queue_hyp_p", torch.randn(dim, K))
        self.queue_hyp_p = nn.functional.normalize(self.queue_hyp_p, dim=0) 
        self.register_buffer("queue_p", torch.randn(dim, K))
        self.queue_p = nn.functional.normalize(self.queue_p, dim=0)

        self.register_buffer("queue_ptr_c", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_hyp_p", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_p", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_c(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_c)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_c[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_c[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_hyp_p(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_hyp_p)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_hyp_p[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_hyp_p[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue_p(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_p)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_p[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_p[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def decouple_norm_from_direction(self, v):
        v_hat = nn.functional.normalize(v, dim=1).clone().detach()
        v_norm = v.norm(2, dim=1, keepdim=True)
        return v_hat*v_norm

    def forward(self, im_euc_p, im_euc_c, im_hyp_p, im_hyp_c):
        """
        Input:
            im_p: a batch of parent images
            im_c: a batch of children images
        Output:
            logits, targets
        """
        logits_e = logits_h = labels_e = labels_h = None
        N, C, H, W = im_euc_p.shape

        # compute query features
        q_euc_p = self.encoder_q(im_euc_p)  # queries: NxC
        q_euc_c = self.encoder_q(im_euc_c)  # queries: NxC

        # moco
        qp_normalized = nn.functional.normalize(q_euc_p, dim=1)
        qc_normalized = nn.functional.normalize(q_euc_c, dim=1)

        if self.theta > 0:
            # q_hyp_p = self.encoder_q(im_hyp_p)  # queries: NxC
            q_hyp_c = self.encoder_q(im_hyp_c)  # queries: NxC

            # hyperbolic
            if self.decoupled:
                q_hyp_c_Rn = self.decouple_norm_from_direction(q_hyp_c)
            else:
                q_hyp_c_Rn = q_hyp_c
            qc_proj = self.tp(q_hyp_c_Rn) # Nx1

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_euc_p, idx_unshuffle_p = self._batch_shuffle_ddp(im_euc_p)
            im_euc_c, idx_unshuffle_c = self._batch_shuffle_ddp(im_euc_c)

            k_euc_p = self.encoder_k(im_euc_p)  # keys: NxC
            k_euc_c = self.encoder_k(im_euc_c)  # keys: NxC

            # moco
            kp_normalized = nn.functional.normalize(k_euc_p, dim=1)
            kp_normalized = self._batch_unshuffle_ddp(kp_normalized, idx_unshuffle_p)
            kc_normalized = nn.functional.normalize(k_euc_c, dim=1)
            kc_normalized = self._batch_unshuffle_ddp(kc_normalized, idx_unshuffle_c)

            if self.theta > 0:
                im_hyp_p, idx_unshuffle_hyp_p = self._batch_shuffle_ddp(im_hyp_p)
                k_hyp_p = self.encoder_k(im_hyp_p)  # keys: NxC

                # hyperbolic
                kp_proj= self.tp(k_hyp_p)
                kp_proj = self._batch_unshuffle_ddp(kp_proj, idx_unshuffle_hyp_p)


        # moco
        l_pos_e = torch.einsum('nc,nc->n', [qp_normalized, kc_normalized]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [qp_normalized, self.queue_c.clone().detach()])
        l_pos_e2 = torch.einsum('nc,nc->n', [qc_normalized, kp_normalized]).unsqueeze(-1)
        l_neg_e2 = torch.einsum('nc,ck->nk', [qc_normalized, self.queue_p.clone().detach()])

        # logits: Nx(1+K)
        logits_e = torch.cat([torch.cat([l_pos_e, l_neg_e], dim=1), torch.cat([l_pos_e2, l_neg_e2], dim=1)], 0)

        # apply temperature
        logits_e /= self.T


        # labels: positive key indicators
        labels_e = torch.zeros(logits_e.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_p(kp_normalized)
        self._dequeue_and_enqueue_c(kc_normalized)

        # hyperbolic
        if self.theta > 0:
            l_pos_h = -self.hyperbolic_dis(qc_proj, kp_proj)
            l_neg_h = -self.hyperbolic_dis(qc_proj.unsqueeze(1), self.queue_hyp_p.clone().detach().T.unsqueeze(0)).squeeze()
            logits_h = torch.cat([l_pos_h, l_neg_h], dim=1)
            logits_h /= self.T
            labels_h = torch.zeros(logits_h.shape[0], dtype=torch.long).cuda()
            self._dequeue_and_enqueue_hyp_p(kp_proj)
        # moco
        return logits_e, logits_h, labels_e, labels_h

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
