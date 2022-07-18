from e2cnn import nn, gspaces
import torch

class EquivariantRewardModelDihedral(torch.nn.Module):
    def __init__(self, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.n_hidden = n_hidden
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.conv = nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr] + 2 * [self.d4_act.trivial_repr] + 1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]),
                              nn.FieldType(self.d4_act, 2 * [self.d4_act.trivial_repr]),
                              kernel_size=1, padding=0, initialize=initialize)

    def forward(self, latent_state, act):
        batch_size = latent_state.shape[0]
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        cat = torch.cat((latent_state.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1), dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.d4_act, self.n_hidden * [self.d4_act.regular_repr] + n_inv * [self.d4_act.trivial_repr] + 1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]))
        out = self.conv(cat_geo)
        return out.tensor.reshape(batch_size, 2)

class EquivariantTransitionModelDihedral(torch.nn.Module):
    def __init__(self, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.n_hidden = n_hidden
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.conv = nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr] + 2 * [self.d4_act.trivial_repr] + 1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]),
                              nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                              kernel_size=1, padding=0, initialize=initialize)

    def forward(self, latent_state, act):
        batch_size = latent_state.shape[0]
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        cat = torch.cat((
                        latent_state.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1),
                        dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.d4_act,
                                                       self.n_hidden * [self.d4_act.regular_repr] + n_inv * [
                                                           self.d4_act.trivial_repr] + 1 * [
                                                           self.d4_act.irrep(1, 1)] + 1 * [
                                                           self.d4_act.quotient_repr((None, 4))]))
        out = self.conv(cat_geo)
        return out.tensor.reshape(batch_size, -1)