import numpy as np
import time

# Assuming you have a list of instances to run
instances_list = [inst1, inst2, inst3]  # Replace with your actual instances
results_all_instances = []
inference_time_all_instances = []

for inst in instances_list:
    env2 = JsspN5(n_job=p_j, n_mch=p_m, low=p_l, high=p_h, reward_type='yaoxin', fea_norm_const=fea_norm_const)

    print('Starting rollout DRL policy...')
    chunk_size = inst.shape[0]

    n_chunks = inst.shape[0] // chunk_size
    results_each_init, inference_time_each_init = [], []

    chunk_result, chunk_time = [], []
    inst_chunk = inst[0:chunk_size]

    batch_data = BatchGraph()

    # states  =   x, edge_indices_pc, edge_indices_mc, batch
    states, feasible_actions, _ = env2.reset(instances=inst_chunk, init_type=init, device=dev, plot=show)
    drl_start = time.time()

    while env2.itr < cap_horizon:
        batch_data.wrapper(*states)
        actions, _ = policy(batch_data, feasible_actions)
        states, _, feasible_actions, _ = env2.step(actions, dev, plot=show)

        # t2 = time.time()
        for log_horizon in performance_milestones:
            if env2.itr == log_horizon:
                DRL_result = env2.incumbent_objs.cpu().squeeze().numpy()
                chunk_result.append(DRL_result)
                chunk_time.append(time.time() - drl_start)

                if n_chunks == 1:
                    print('For testing steps: {}    '.format(env2.itr), 'DRL Gap: {:.6f}    '.format(
                        ((DRL_result - gap_against) / gap_against).mean()),
                        'DRL results takes: {:.6f} per instance.'.format(
                            (time.time() - drl_start) / chunk_size))

    results_each_init.append(np.stack(chunk_result))
    inference_time_each_init.append(np.array(chunk_time))
    results_each_init = np.concatenate(results_each_init, axis=-1)
    inference_time_each_init = ((np.stack(inference_time_each_init).sum(axis=0)) / n_chunks) / chunk_size

    # Store results for this instance
    results_all_instances.append(results_each_init)
    inference_time_all_instances.append(inference_time_each_init)

# Compare results across all instances
for i, (results, times) in enumerate(zip(results_all_instances, inference_time_all_instances)):
    print(f"Instance {i+1}:")
    print(f"Results: {results}")
    print(f"Inference Times: {times}")