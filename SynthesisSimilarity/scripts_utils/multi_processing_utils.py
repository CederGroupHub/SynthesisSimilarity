import json
import math
import os
import time
from pprint import pprint
import numpy as np
import multiprocessing as mp
import argparse
import sys

import concurrent.futures

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def run_multiprocessing_tasks(
    tasks,
    thread_func,
    func_args=(),
    num_cores=4,
    verbose=False,
    join_results=False,
    use_threading=False,
    mp_context=None,
):
    # execute pipeline in a parallel way
    last_time = time.time()

    # get parallel_arguments
    if tasks:
        parallel_arguments = []
        num_tasks_per_core = math.ceil(len(tasks)/num_cores)
        for i in range(num_cores):
            parallel_arguments.append(
                (tasks[i*num_tasks_per_core: (i+1)*num_tasks_per_core], ) + func_args
            )
    else:
        parallel_arguments = [func_args] * num_cores

    if not use_threading:
        # running using mp
        # use 'spawn' for tf
        mp_ctx = mp.get_context(mp_context)
        p = mp_ctx.Pool(processes=num_cores)
        all_summary = p.starmap(thread_func, parallel_arguments)
        p.close()
        p.join()
    else:
        # running using threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures =[
                executor.submit(thread_func, *(parallel_arguments[i]))
                for i in range(num_cores)
            ]
            all_summary = [f.result() for f in futures]

    if verbose:
        # TODO: maybe the type of tmp_summary here is not very correct
        # reading results
        print('time used:', time.time()-last_time)
        if isinstance(all_summary[0], dict) and 'success_tasks' in all_summary[0]:
            # combine all results
            all_success_tasks = sum([tmp_summary['success_tasks'] for tmp_summary in all_summary], [])
            print('len(all_success_tasks)', len(all_success_tasks))

        if isinstance(all_summary[0], dict) and 'error_tasks' in all_summary[0]:
            # combine all error tasks
            all_error_tasks = sum([tmp_summary['error_tasks'] for tmp_summary in all_summary], [])
            print('len(all_error_tasks)', len(all_error_tasks))

    if join_results and isinstance(all_summary[0], list):
        # when output is a single variable, the mp output is a list with length of cores,
        #   where each element is a list of results from each processor.
        #   Therefore, need to sum to combine results
        last_results = sum(all_summary, [])
    elif join_results and isinstance(all_summary[0], tuple):
        # when output is multiple variables (a tuple), the mp output is a tuple,
        #   where each variable is a list of results from each processor.
        #   Therefore, need to sum to combine results in each variable
        last_results = []
        for i in range(len(all_summary[0])):
            last_results.append([x[i] for x in all_summary])
    else:
        last_results = all_summary

    return last_results

def save_results(results, dir_path='../generated/results', prefix='results'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = '{}_{}.json'.format(prefix, str(hash(str(results))))
    with open(os.path.join(dir_path, file_name), 'w') as fw:
        json.dump(results, fw, indent=2)
    return file_name
