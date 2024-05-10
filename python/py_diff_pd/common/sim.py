import torch
import torch.nn as nn
import torch.autograd as autograd

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray


class SimFunction(autograd.Function):

    @staticmethod
    def forward(ctx, deformable, dofs, act_dofs, method, q, v, a, f_ext, dt, option):

        ctx.deformable = deformable
        ctx.dofs = dofs
        ctx.act_dofs = act_dofs
        ctx.method = method
        ctx.dt = dt
        ctx.option = option

        q_array = StdRealVector(q.detach().numpy())
        v_array = StdRealVector(v.detach().numpy())
        a_array = StdRealVector(a.detach().numpy())
        f_ext_array = StdRealVector(f_ext.detach().numpy())
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)

        deformable.PyForward(
            method, q_array, v_array, a_array, f_ext_array, dt, option, q_next_array, v_next_array, StdIntVector(0))

        # Store for backward
        q_next = torch.as_tensor(ndarray(q_next_array))
        v_next = torch.as_tensor(ndarray(v_next_array))
        ctx.mat_w_dofs = deformable.NumOfPdElementEnergies()
        ctx.act_w_dofs = deformable.NumOfPdMuscleEnergies()
        ctx.state_p_dofs = deformable.NumOfStateForceParameters()

        ctx.save_for_backward(q, v, a, f_ext, q_next, v_next)

        return q_next, v_next

    @staticmethod
    def backward(ctx, dl_dq_next, dl_dv_next):

        q, v, a, f_ext, q_next, v_next = ctx.saved_tensors
        dofs, act_dofs, mat_w_dofs, act_w_dofs, state_p_dofs = ctx.dofs, ctx.act_dofs, ctx.mat_w_dofs, ctx.act_w_dofs, ctx.state_p_dofs

        q_array = StdRealVector(q.detach().numpy())
        v_array = StdRealVector(v.detach().numpy())
        a_array = StdRealVector(a.detach().numpy())
        f_ext_array = StdRealVector(f_ext.detach().numpy())
        q_next_array = StdRealVector(q_next.detach().numpy())
        v_next_array = StdRealVector(v_next.detach().numpy())

        dl_dq_next_array = StdRealVector(dl_dq_next.detach().numpy())
        dl_dv_next_array = StdRealVector(dl_dv_next.detach().numpy())

        dl_dq_array = StdRealVector(dofs)
        dl_dv_array = StdRealVector(dofs)
        dl_da_array = StdRealVector(act_dofs)
        dl_df_ext_array = StdRealVector(dofs)
        dl_dmat_wi = StdRealVector(mat_w_dofs)
        dl_dact_wi = StdRealVector(act_w_dofs)
        dl_dstate_pi = StdRealVector(state_p_dofs)

        ctx.deformable.PyBackward(
            ctx.method, q_array, v_array, a_array, f_ext_array, ctx.dt,
            q_next_array, v_next_array, StdIntVector(0), dl_dq_next_array, dl_dv_next_array, ctx.option,
            dl_dq_array, dl_dv_array, dl_da_array, dl_df_ext_array, dl_dmat_wi, dl_dact_wi, dl_dstate_pi)

        dl_dq = torch.as_tensor(ndarray(dl_dq_array))
        dl_dv = torch.as_tensor(ndarray(dl_dv_array))
        dl_da = torch.as_tensor(ndarray(dl_da_array))
        dl_df_ext = torch.as_tensor(ndarray(dl_df_ext_array))

        return (None, None, None, None,
            torch.as_tensor(ndarray(dl_dq)),
            torch.as_tensor(ndarray(dl_dv)),
            torch.as_tensor(ndarray(dl_da)),
            torch.as_tensor(ndarray(dl_df_ext)),
            None, None)



class Sim(nn.Module):
    def __init__(self, deformable):
        super(Sim, self).__init__()
        self.deformable = deformable

    def forward(self, dofs, act_dofs, method, q, v, a, f_ext, dt, option):
        return SimFunction.apply(
            self.deformable, dofs, act_dofs, method, q, v, a, f_ext, dt, option)