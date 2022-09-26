from tqdm import tqdm
import os
import sys

sys.path.append('..')

from rot_est.parameters import *
from rot_est.env_wrapper import EnvWrapper
from rot_est.logger import Logger

import torch
import torch.nn.functional as F
from e2cnn import gspaces, nn
from rot_est.equi_net import Equivariant
from rot_est.cnn_net import CNN
from rot_est.torch_utils import randomCrop, centerCrop

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collectData():
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

    n_data = 1000
    buffer = []
    pbar = tqdm(total=n_data)
    while len(buffer) < n_data:
        state, obs = envs.reset()
        for i in range(num_processes):
            buffer.append((obs[i], state[i]))
            pbar.update(1)
            if len(buffer) >= n_data:
                break

    torch.save(buffer, 'pose_{}_{}_{}_{}.pt'.format(env, corrupt, heightmap_size, n_data))
    del buffer

def train(dataset):
    min_epochs = 100
    max_epochs_no_improve = 100
    max_epochs = 1000
    n_holdout = 200
    if model == 'equi_c':
        group = gspaces.Rot2dOnR2(8)
        network = Equivariant(group, 8, 64).to(device)
    elif model == 'cnn_sim':
        network = CNN(obs_shape=(8, 128, 128), out_dim=8).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    log_dir = os.path.join(log_pre, 'equi_map_{}'.format(model)) + '{}_aug{}_ndata{}'.format(corrupt, aug, n_data)
    logger = Logger(log_dir, env, 'train', num_processes, 1000, 1, log_sub)
    hyper_parameters['model_shape'] = str(network)
    logger.saveParameters(hyper_parameters)

    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    gc = torch.stack([dataset[i][1] for i in range(len(dataset))]).long()
    data_permutation = np.random.permutation(len(data))
    data = data[data_permutation]
    gc = gc[data_permutation]
    train_data = data[:n_data]
    train_gc = gc[:n_data]
    valid_data = data[n_data:n_data+n_holdout]
    valid_gc = gc[n_data:n_data+n_holdout]
    test_data = data[n_data+n_holdout:n_data+2*n_holdout]
    test_gc = gc[n_data+n_holdout:n_data+2*n_holdout]

    if n_img_trans > 0:
        trans_data = []
        trans_gc = []
        for i in range(n_img_trans):
            c8 = gspaces.Rot2dOnR2(8)
            o0 = train_data[i:i+1]
            o0 = nn.GeometricTensor(o0, nn.FieldType(c8, 8 * [c8.trivial_repr]))
            for tran in c8.testing_elements:
                trans_data.append(o0.transform(tran).tensor)
                trans_gc.append(train_gc[i:i+1])
        trans_data = torch.cat(trans_data)
        trans_gc = torch.cat(trans_gc)
        train_data = torch.cat([train_data, trans_data], 0)
        train_gc = torch.cat([train_gc, trans_gc], 0)

    min_valid_loss = 1e10
    epochs_no_improve = 0
    min_test_err = 1e10

    pbar = tqdm(total=max_epochs)
    for epoch in range(1, max_epochs+1):
        train_idx = np.random.permutation(train_data.shape[0])
        if no_bar:
            it = range(0, train_data.shape[0], batch_size)
        else:
            it = tqdm(range(0, train_data.shape[0], batch_size))
        for start_pos in it:
            idx = train_idx[start_pos: start_pos + batch_size]
            batch = train_data[idx]
            gc_batch = train_gc[idx]
            batch = batch.to(device)
            label = gc_batch.to(device)
            if aug:
                # aug_true = []
                # for j in range(batch.shape[0]):
                #     aug_true.append(randomCrop(batch[j:j + 1], out=crop_size))
                # batch = torch.cat(aug_true)
                batch = randomCrop(batch, out=crop_size)
            elif heightmap_size > crop_size:
                batch = centerCrop(batch, out=crop_size)
            out = network(batch)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.model_losses.append(loss.item())

        with torch.no_grad():
            network.eval()
            valid_data = valid_data.to(device)
            if heightmap_size > crop_size:
                valid_data = centerCrop(valid_data, out=crop_size)
            valid_out = network(valid_data)
            # valid_loss = F.cross_entropy(valid_out, valid_gc.to(device))
            valid_loss = 1 - (valid_out.argmax(1) == valid_gc.to(device)).sum() / valid_out.shape[0]
            valid_loss = valid_loss.item()

            test_data = test_data.to(device)
            if heightmap_size > crop_size:
                test_data = centerCrop(test_data, out=crop_size)
            test_out = network(test_data)
            acc = (test_out.argmax(1) == test_gc.to(device)).sum() / test_out.shape[0]
            test_err = acc.item()
            network.train()
        logger.model_holdout_losses.append((valid_loss, test_err))
        logger.saveModelLosses()
        # logger.saveModelLossCurve()
        # logger.saveModelHoldoutLossCurve()
        # if epoch % 10 == 0:
        #     torch.save(network.state_dict(), os.path.join(logger.models_dir, 'model_{}.pt'.format(epoch)))
            # torch.save(contrastive.state_dict(), os.path.join(logger.models_dir, 'contrastive_{}.pt'.format(epoch)))

        if valid_loss < min_valid_loss:
            epochs_no_improve = 0
            min_valid_loss = valid_loss
            min_test_err = test_err
        else:
            epochs_no_improve += 1
        if not no_bar:
            pbar.set_description('epoch: {}, valid loss: {:.03f}, no improve: {}, test err: {:.03f} ({:.03f})'
                                 .format(epoch, valid_loss, epochs_no_improve, test_err, min_test_err))
            pbar.update()
        if epochs_no_improve >= max_epochs_no_improve and epoch > min_epochs:
            break
    pbar.close()
    logger.saveModelLossCurve()
    logger.saveModelHoldoutLossCurve()
    del network

if __name__ == '__main__':
    # group = gspaces.FlipRot2dOnR2(4)
    # network = EquivariantEncoder128(group, 4, 64, n_reg=3).to(device)
    # print('equi: {}'.format(sum(p.numel() for p in network.parameters() if p.requires_grad)))
    # network = SACEncoderFullyConvSimilarNParameter(obs_shape=(4, 128, 128), out_dim=1 * 8)
    # print('cnn_fully_conv_sim: {}'.format(sum(p.numel() for p in network.parameters() if p.requires_grad)))
    # print(1)

    # global corrupt
    # # for corrupt in [[''], ['grid'], ['occlusion'], ['random_light_color'], ['light_effect'], ['two_light_color'],
    # #                 ['two_specular'], ['squeeze'], ['reflect'], ['reflect', 'grid'], ['random_reflect'],
    # #                 ['random_reflect', 'grid'], ['side'], ['side' ,'grid'], ['side', 'light_effect'],
    # #                 ['side', 'squeeze'], ['side', 'reflect'], ['side', 'reflect', 'grid'], ['side', 'random_reflect'],
    # #                 ['side', 'random_reflect', 'grid']]:
    # for corrupt in [['side']]:
    #     env_config['corrupt'] = corrupt
    #     env_config['seed'] = 0
    #     collectData()
    #
    # from scripts.main import set_seed
    # global seed
    # global note
    # for n in (500, 200,):
    #     for seed in range(0, 4):
    #         note = '{}_aug{}_ndata{}'.format(corrupt, aug, n)
    #         set_seed(seed)
    #         train(n_data=n)
    #         torch.cuda.empty_cache()

    # plt.figure(dpi=300)
    # # all_corrupt = [[''], ['grid'], ['occlusion'], ['random_light_color'], ['light_effect'], ['two_light_color'],
    # #                              ['two_specular'], ['squeeze'], ['reflect'], ['condition_reverse'],
    # #                              ['side'], ['side' ,'grid'], ['side', 'light_effect'],
    # #                              ['side', 'squeeze'], ['side', 'reflect']]
    #
    # all_corrupt = [['side', 'grid']]
    # # all_corrupt = [[''], ['grid'], ['occlusion'], ['light_effect'], ['squeeze'], ['reflect'], ['condition_reverse'],
    # #                ['side']]
    #
    # obss = []
    # for i, corrupt in enumerate(all_corrupt):
    #     env_config['corrupt'] = corrupt
    #     env_config['seed'] = 3
    #     envs = EnvWrapper(1, simulator, env, env_config, planner_config)
    #     obs = envs.reset()[1]
    #     obss.append(obs[0][:3].permute(1, 2, 0))
    #
    # fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    # axs = axs.reshape(-1)
    # for i in range(len(all_corrupt)):
    #     corrupt = all_corrupt[i]
    #     axs[i].imshow(obss[i])
    #     axs[i].axis(False)
    #     axs[i].set_title((','.join(corrupt) if corrupt != [''] else 'no corrupt').replace('_', ' '), fontsize=30)
    #     print(i)
    # for j in range(i+1, axs.shape[0]):
    #     axs[j].axis('off')
    # plt.tight_layout()
    # plt.show()

    # for corrupt in [[''], ['grid'], ['occlusion'], ['random_light_color'], ['light_effect'], ['two_light_color'],
    #                 ['two_specular'], ['squeeze'], ['reflect'], ['reflect', 'grid'], ['random_reflect'],
    #                 ['random_reflect', 'grid'], ['side'], ['side' ,'grid'], ['side', 'light_effect'],
    #                 ['side', 'squeeze'], ['side', 'reflect'], ['side', 'reflect', 'grid'], ['side', 'random_reflect'],
    #                 ['side', 'random_reflect', 'grid']]:
    #     dataset = torch.load(os.path.join(load_buffer, '{}_{}_{}_1000.pt'.format(env, corrupt, heightmap_size)))
    #     idx = np.random.randint(1000)
    #     true_obs = dataset[idx][0]
    #     trans_obs = dataset[idx][1]
    #     fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    #     axs = axs.reshape(-1)
    #     for i in range(8):
    #         axs[i].imshow(true_obs[i, :3].permute(1, 2, 0).cpu())
    #     plt.tight_layout()
    #     fig.show()
    #
    #     fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    #     axs = axs.reshape(-1)
    #     for i in range(8):
    #         axs[i].imshow(trans_obs[i, :3].permute(1, 2, 0).cpu())
    #     plt.tight_layout()
    #     fig.show()

    dataset = torch.load('trans_est_C8_{}_{}_{}_2000.pt'.format(env, corrupt, heightmap_size))
    # dataset = torch.load(os.path.join(load_buffer, 'absolute_pose_{}_{}_{}_10000.pt'.format(env, corrupt, heightmap_size)))
    # for corrupt in [[''], ['grid'], ['side'], ['side', 'grid'], ['occlusion'], ['random_light_color']]:
    for s in range(0, 4):
        args.seed = s
        set_seed(s)
        train(dataset)
        torch.cuda.empty_cache()

    # from scripts.main import set_seed
    # global seed
    # global note
    # global corrupt
    # for corrupt in [[''], ['grid'], ['side']]:
    #     for n in (500, 200,):
    #         for seed in range(0, 4):
    #             note = '{}_aug{}_ndata{}'.format(corrupt, aug, n)
    #             set_seed(seed)
    #             train(n_data=n, max_epochs=100)
    #             torch.cuda.empty_cache()

