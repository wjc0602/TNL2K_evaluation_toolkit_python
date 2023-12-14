import os


def config_sequence(type, seq_eval_config):
    # config sequences for evaluation
    # the configuration files are placed in ./sequence_evaluation_config/
    if type == 'TNL2K_all':
        dataset_name = os.path.join(seq_eval_config, 'TNL2K_ALL.txt')
    elif type == 'TNL2K_testing_set':
        dataset_name = os.path.join(seq_eval_config, 'TNL2K_testing_set.txt')
    elif type == 'TNL2K_50':
        dataset_name = os.path.join(seq_eval_config, 'TNL2K_50.txt')
    else:
        raise ValueError('Error in evaluation dataset type!')

    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f'{dataset_name} is not found!')

    # load evaluation sequences
    with open(dataset_name, 'r') as file:
        sequences = file.readlines()

    sequences = [seq.strip() for seq in sequences]
    return sequences
