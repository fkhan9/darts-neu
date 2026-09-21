"""
Second-order (unrolled) DARTS Architect, following Liu et al. 2018.

Approximates dL_val/dalpha at the one-step-unrolled weights w' (rather
than the current weights w), using a finite-difference approximation
of the Hessian-vector product to avoid computing the true second
derivative explicitly. This is the "second-order approximation"
described in the original DARTS paper (Eq. 7-8), as opposed to the
cheaper first-order approximation (which just treats w as fixed and
ignores the w'(alpha) dependency entirely).

Costs roughly 2-3x the compute/memory of first-order per architecture
step (one unrolled train step, one val backward pass on the unrolled
model, plus two finite-difference passes over a train batch for the
Hessian-vector product). Verify this fits your GPU memory budget
before committing to a full run -- if not, first-order is a legitimate,
documentable fallback given real resource constraints.
"""

import copy
import torch


class Architect:
    def __init__(self, model, arch_lr=3e-4, arch_weight_decay=1e-3,
                 unrolled_momentum=0.9, unrolled_weight_decay=3e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=arch_lr,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )
        # must match the weight optimizer's momentum/weight_decay for the
        # unrolled virtual step to correctly approximate one real SGD step
        self.unrolled_momentum = unrolled_momentum
        self.unrolled_weight_decay = unrolled_weight_decay

    def step(self, x_train, y_train, x_val, y_val, eta, w_optimizer, criterion):
        self.optimizer.zero_grad()
        loss_value = self._backward_step_unrolled(x_train, y_train, x_val, y_val, eta, w_optimizer, criterion)
        self.optimizer.step()
        return loss_value

    # ---- internals ----

    def _compute_unrolled_model(self, x_train, y_train, eta, w_optimizer, criterion):
        logits = self.model(x_train)
        loss = criterion(logits, y_train)

        theta = torch.cat([p.data.view(-1) for p in self.model.parameters()])

        try:
            moment = torch.cat([
                w_optimizer.state[p].get('momentum_buffer', torch.zeros_like(p)).view(-1)
                for p in self.model.parameters()
            ])
            moment = moment.mul_(self.unrolled_momentum)
        except Exception:
            moment = torch.zeros_like(theta)

        dtheta = torch.cat([
            (g + self.unrolled_weight_decay * p).view(-1)
            for g, p in zip(
                torch.autograd.grad(loss, self.model.parameters()),
                self.model.parameters(),
            )
        ])

        unrolled_theta = theta.sub(moment + dtheta, alpha=eta)
        unrolled_model = self._construct_model_from_theta(unrolled_theta)
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = copy.deepcopy(self.model)
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for name, p in self.model.named_parameters():
            v_length = p.numel()
            params[name] = theta[offset: offset + v_length].view(p.size())
            offset += v_length
        assert offset == len(theta)

        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(theta.device)

    def _backward_step_unrolled(self, x_train, y_train, x_val, y_val, eta, w_optimizer, criterion):
        unrolled_model = self._compute_unrolled_model(x_train, y_train, eta, w_optimizer, criterion)
        unrolled_logits = unrolled_model(x_val)
        unrolled_loss = criterion(unrolled_logits, y_val)

        unrolled_loss.backward()

        dalpha = [v.grad.data for v in unrolled_model.arch_parameters()]
        dtheta = [v.grad.data for v in unrolled_model.parameters()]

        implicit_grads = self._hessian_vector_product(dtheta, x_train, y_train, criterion)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.clone()
            else:
                v.grad.data.copy_(g.data)

        # free the unrolled model's graph explicitly -- this is the step
        # most likely to be missing/wrong if second-order OOMs in practice
        loss_value = unrolled_loss.item()
        del unrolled_model, unrolled_logits, unrolled_loss
        return loss_value

    def _hessian_vector_product(self, vector, x_train, y_train, criterion, r=1e-2):
        R = r / torch.cat([v.view(-1) for v in vector]).norm()

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = criterion(self.model(x_train), y_train)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = criterion(self.model(x_train), y_train)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)  # restore original weights

        return [(p - n).div_(2 * R) for p, n in zip(grads_p, grads_n)]