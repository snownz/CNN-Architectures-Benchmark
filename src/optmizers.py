import torch
import torch.optim as optim

# https://paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1
class SAM(optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
    
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
    
        defaults = dict( rho = rho, **kwargs )
        super(SAM, self).__init__( params, defaults )
        self.base_optimizer = base_optimizer( self.param_groups, **kwargs )
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / ( grad_norm + 1e-12 )
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['e_w'] = p.grad * scale
                p.add_(self.state[p]['e_w'])  # ascent step

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # descent step

        self.base_optimizer.step()  # apply optimizer step

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        
        shared_device = self.param_groups[0]["params"][0].device
        
        norm = torch.norm(
            torch.stack( [
                p.grad.norm( p = 2 ).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ] ),
            p = 2
        )
        return norm