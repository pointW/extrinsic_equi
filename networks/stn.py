import torch
import torch.nn.functional as F

def linspace_from_neg_one(num_steps):
    r = torch.linspace(-1, 1, num_steps)
    r = r * (num_steps - 1) / num_steps
    return r

def perspective_grid_generator(theta, size):
    n,c,h,w = size
    grid = torch.zeros(n,h,w,3, device=theta.device)
    grid.select(-1, 0).copy_(linspace_from_neg_one(w))
    grid.select(-1, 1).copy_(linspace_from_neg_one(h).unsqueeze(-1))
    grid.select(-1, 2).fill_(1)
    grid = grid.view(n,h*w,3) @ theta.transpose(1,2)
    grid = grid.view(n,h,w,3)
    grid = grid[:,:,:,:2] / grid[:,:,:,2:3]
    return grid

class STN(torch.nn.Module):
    def __init__(self, obs_shape=(1, 64, 64)):
        super().__init__()
        # self.localization = torch.nn.Sequential(
        #     torch.nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(2),
        #     # 32x32
        #     torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(2),
        #     # 16x16
        #     torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(2),
        #     # 8x8
        #     torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(128, 128, kernel_size=3),
        #     torch.nn.ReLU(inplace=True),
        #     # 6x6
        #     torch.nn.MaxPool2d(2),
        #     # 3x3
        # )
        #
        # # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(128 * 3 * 3, 512),
        #     torch.nn.ReLU(True),
        #     torch.nn.Linear(512, 8)
        # )

        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # 32x32
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # 16x16
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # 8x8
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # 4x4
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.fc_loc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 8)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(torch.moveaxis(x[0, :3], 0, 2).cpu())

        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = torch.cat((theta, torch.ones(x.shape[0], 1, device=x.device)), dim=1)
        theta = theta.view(-1, 3, 3)

        grid = perspective_grid_generator(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=False)

        # axs[1].imshow(torch.moveaxis(x[0, :3], 0, 2).cpu())
        # fig.show()
        return x

class STN2(torch.nn.Module):
    def __init__(self, obs_shape=(1, 64, 64)):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(torch.moveaxis(x[0, :3], 0, 2).cpu())

        theta = self.theta.unsqueeze(0).expand(x.shape[0], self.theta.shape[0])
        theta = torch.cat((theta, torch.ones(x.shape[0], 1, device=x.device)), dim=1)
        theta = theta.view(-1, 3, 3)

        grid = perspective_grid_generator(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=False)

        # axs[1].imshow(torch.moveaxis(x[0, :3], 0, 2).cpu())
        # fig.show()
        return x
