# -*- coding: utf-8 -*-
import wandb

PROJECT_URL = 'koulanurag/pytorch-drl'
from policybazaar.config import MAX_PRE_TRAINED_LEVEL, MIN_PRE_TRAINED_LEVEL


def get_env_info():
    api = wandb.Api()

    env_info = {}
    for run in [r for r in api.runs(PROJECT_URL) if r.config['observation_noise_std'] == 0.01]:
        env_name = run.config['case'] + ':' + run.config['env_name']
        env_info[env_name] = {'wandb_run_id': PROJECT_URL + '/' + run.id}
        history = run.scan_history()
        env_info[env_name]['info'] = {}

        for row in history:
            if 'save_interval' in row:
                pre_train_marker = 'pre_trained={}'.format(MAX_PRE_TRAINED_LEVEL + 1 - row['save_interval'])
                env_info[env_name]['info'][pre_train_marker] = {'score_mean': round(row['interval_test_score'], 1),
                                                                'score_std': round(row["interval_test_score_std"], 1)}

    return env_info


def markdown_pre_trained_scores(env_info):
    # create markdown for the table:
    msg = "| Environment Name |"
    for i in range(MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL + 1):
        pre_info = '(best)' if i == MIN_PRE_TRAINED_LEVEL else ('(worst)' if i == MAX_PRE_TRAINED_LEVEL else '')
        msg += "`pre_trained={}` {} ".format(i, pre_info) + "|"

    msg += '\n'
    msg += "|" + " | ".join(":------:" for _ in range(MAX_PRE_TRAINED_LEVEL + 1)) + ' | ' + '\n'

    for env_name in env_info:
        msg += "|{}|".format("`{}`".format(env_name))
        for i in range(MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL + 1):
            _key = 'pre_trained={}'.format(i)
            if _key in env_info[env_name]['info']:
                pre_trained_info = env_info[env_name]['info'][_key]
                msg += '{}±{} |'.format(pre_trained_info['score_mean'], pre_trained_info['score_std'])
            else:
                msg += ' |'
        msg += '\n'
    return msg


if __name__ == '__main__':
    env_info = get_env_info()
    print(env_info)

    table_markdown = markdown_pre_trained_scores(env_info)
    print(table_markdown)
