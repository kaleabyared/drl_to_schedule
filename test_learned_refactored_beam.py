import numpy as np
import os
import torch
import time
import copy

import random
from env.environment import JsspN5, BatchGraph
from model.policy import Actor
from ortools_solver import MinimalJobshopSat
from pathlib import Path


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)  # bug, refer to https://github.com/pytorch/pytorch/issues/61032

    show = False
    force_test = True

    file_sufix = "run_bgrd_001"

    save = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if dev == 'cuda': print('Using UniTS CUDA to test...')
    else: print('Using UniTS CPU to test...')

    beam_width = 6
    is_gready_beam = False


    # benchmark config
    p_l = 1
    p_h = 99
    init_type = 'fdd-divide-mwkr'  # ['fdd-divide-mwkr', 'spt']

    test_t = 'tai'   #, 'abz', 'orb', 'yn', 'swv', 'la', 'ft']  # ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft', 'syn']

    # Not to exceed 100x20 for chunk size 1
    tai_problem_j = [15, 20, 20] #, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20] #, 20, 15, 20, 15, 20, 20]

    problem_j, problem_m = tai_problem_j, tai_problem_m

    # model config
    model_j = 10  # 10， 15， 15， 20， 20
    model_m = 10  # 10， 10， 15， 10， 15
    model_l = 1
    model_h = 99
    model_init_type = 'fdd-divide-mwkr'
    reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
    gamma = 1

    hidden_dim = 128
    embedding_layer = 4
    policy_layer = 4
    embedding_type = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
    heads = 1
    drop_out = 0.

    lr = 5e-5  # 5e-5, 4e-5
    steps_learn = 10
    training_episode_length = 500
    batch_size = 64
    episodes = 128000  # 128000, 256000
    step_validation = 10

    model_type = 'incumbent'  # 'incumbent', 'last-step'

    if embedding_type == 'gin':
        dghan_param_for_saved_model = 'NAN'
    elif embedding_type == 'dghan' or embedding_type == 'gin+dghan':
        dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)
    else:
        raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    # MDP config
    cap_horizon = 5000
    performance_milestones = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000]
    result_type = 'incumbent'  # 'current', 'incumbent'
    fea_norm_const = 1000


    for p_j, p_m in zip(problem_j, problem_m):  # select problem size

        inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))
        print('\nStart testing {}{}x{}...'.format(test_t, p_j, p_m))

        # read saved gap_against
        ortools_path = Path('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
        if ortools_path.is_file():
            gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
        else:
            raise Exception('Result file not found!!!!!')


        torch.manual_seed(seed)
        policy = Actor(in_dim=3, hidden_dim=hidden_dim, embedding_l=embedding_layer, policy_l=policy_layer, embedding_type=embedding_type,
                        heads=heads, dropout=drop_out, is_gready_beam=is_gready_beam, beam_width=beam_width).to(dev)


        saved_model_path = './saved_model/{}_{}x{}[{},{}]_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth' \
            .format(model_type, model_j, model_m, model_l, model_h, model_init_type, reward_type, gamma,
                    hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
                    lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)

        print('Loading model from:', saved_model_path)

        policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

        #Write results to file
        print('Start to test initial solution: {}...'.format(init_type))
        which_model = './test_results/DRL_results/{}_{}x{}[{},{}]_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/' \
            .format(model_type, model_j, model_m, model_l, model_h, model_init_type, reward_type, gamma,
                    hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
                    lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
        which_dateset = '{}_{}x{}_{}_{}'.format(test_t, p_j, p_m, init_type, file_sufix)


        if not Path(which_model).is_dir():
            os.mkdir(which_model)
            with open(which_model + '__init__.py', 'w'): pass  # make created folder as python package by creating __init__.py

        result_file = Path(which_model + which_dateset + '_result.npy')
        time_file = Path(which_model + which_dateset + '_time.npy')

        if not result_file.is_file() or not time_file.is_file() or force_test:
            print('Starting rollout DRL policy...')
            print('For beam search width of: ' + str(beam_width) + " search type is_gready set to: " + str(is_gready_beam))

            chunk_size = inst.shape[0]
            results_each_init, inference_time_each_init = [], []

            # t3 = time.time()
            chunk_result, chunk_time = [], []


            env, batch_data, states, feasible_actions, actions, pi_value = [], [], [], [], [], []
            env_best_result = 999999999

            for i in range(beam_width):
                env.append(JsspN5(n_job=p_j, n_mch=p_m, low=p_l, high=p_h, reward_type=reward_type, fea_norm_const=fea_norm_const))
                batch_data.append(BatchGraph())

                st, fa, _ = env[i].reset(instances=inst, init_type=init_type, device=dev)
                states.append(st)
                feasible_actions.append(fa)

                actions.append([]), pi_value.append([])

            # t4 = time.time()
            drl_start = time.time()
            while env[0].itr < cap_horizon:
                #print(" * " + str(env[0].itr), end="")
                #if (env[0].itr + 1) % 25 == 0: print(" - ")

                # t1 = time.time()

                for i in range(beam_width):
                    #for each beam get three samples (3x3)
                    batch_data[i].wrapper(*states[i])
                    actions[i], pi_value[i] = policy(batch_data[i], feasible_actions[i])

                if env[0].itr == 0:
                    for i in range(beam_width):
                        states[i], _, feasible_actions[i], _ = env[i].step(actions[i][i], dev)

                    pi_weight = np.mean(np.array(pi_value), axis = (1))
                    #print(pi_weight)

                else:
                    #print("Before: ")
                    #print(pi_weight)
                    #print("The New: ")
                    #print(np.mean(np.array(pi_value), axis = (1)))

                    pi_weight *= np.mean(np.array(pi_value), axis = (1))
                    pi_weight /= np.sum(pi_weight)

                    #print("After: ")
                    #print(pi_weight)
                    z = pi_weight.flatten()
                    s_i = np.array(z).argsort()[-beam_width:][::-1]
                    i_d = [[s_i[i] // beam_width, s_i[i] % beam_width] for i in range(beam_width)]
                    #print(z)
                    #print(i_d)

                    env_temp  = []
                    for i in range(beam_width):
                        pi_weight[i] = z[s_i[i]]
                        env_temp.append(copy.deepcopy(env[i_d[i][0]]))

                    #print("Updated ****: ")
                    #print(pi_weight)

                    current_value = env[i_d[0][0]].incumbent_objs.view(-1).cpu().numpy().mean()
                    if current_value < env_best_result:
                        env_best_result = current_value
                        best_env = copy.deepcopy(env[i_d[0][0]])
                        #print("Best value is: ")
                        #print(env_best_result)


                    env = []
                    for i in range(beam_width):
                        #states[i], rewards, feasible_actions[i], dones = env[i].step(actions[i][0], dev)
                        env.append(copy.deepcopy(env_temp[i]))
                        #env[i] = copy.deepcopy(env_temp[i])
                        states[i], _, feasible_actions[i], _ = env[i].step(actions[i_d[i][0]][i_d[i][1]], dev)


                # t2 = time.time()
                for log_horizon in performance_milestones:
                    if env[0].itr == log_horizon:
                        if result_type == 'incumbent':
                            DRL_result = best_env.incumbent_objs.cpu().squeeze().numpy()
                        else:
                            DRL_result = best_env.current_objs.cpu().squeeze().numpy()

                        chunk_result.append(DRL_result)
                        chunk_time.append(time.time() - drl_start)
                        print("")
                        print(DRL_result)
                        print(gap_against)
                        print('For testing steps: {}    '.format(env[0].itr),
                              'DRL Gap: {:.6f}    '.format(((DRL_result - gap_against) / gap_against).mean()),
                              'DRL results takes: {:.6f} per instance.'.format((time.time() - drl_start) / chunk_size))


            results_each_init.append(np.stack(chunk_result))
            inference_time_each_init.append(np.array(chunk_time))


            if save:
                np.save(which_model + which_dateset + '_result.npy', results_each_init)
                np.save(which_model + which_dateset + '_time.npy', inference_time_each_init)

        else:
            print('Results already exist. Not forcing a test!!!')


if __name__ == '__main__':

    main()

