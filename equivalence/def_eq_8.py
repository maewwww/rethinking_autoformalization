import os
import sys
import os.path as osp
import re
import json
import asyncio
import aiohttp
from typing import Set, Dict, List, Optional
import traceback
from copy import deepcopy
import requests
import psutil
import multiprocessing as mp
import regex as re
import collections as C

import aiofiles
import fire
from easydict import EasyDict
from tqdm import tqdm
from loguru import logger

from common.constants import INIT_WAIT_TIME
from common.dataclasses import Environment
from common.repl import REPL


THM_CODE_PATTERN = re.compile(r'theorem (.*?)sorry', flags=re.DOTALL)
CODEBLOCK_PATTERN = re.compile(r'```(?:.*?)\n(.*?)```', flags=re.DOTALL)


def unique(r : list) -> list:
    s = []
    for i in r:
        if i not in s:
            s.append(i)
    return s


def main(
        mathlib_root: str,
        eval_set: str,
        working_root: str,
        dataset_root: str,
        repl_root: str,
        num_concurrency: int=8,
        ):
    saved_args = {**locals()}
    os.makedirs(working_root, exist_ok=True)
    logger.add(osp.join(working_root, 'autoformalization_equiv_checked_def.log'))
    logger.info(f'hyperparameters: {saved_args}')

    samples = []
    with open(osp.join(dataset_root, eval_set, 'benchmark.jsonl'), 'r') as f:
        for line in f.readlines():
            samples.append(json.loads(line))
    # ['informal_stmt', 'formal_stmt', 'header', 'proof_state', 'mathlib_dependencies', 'hard_dependencies', 'source', 'problem_name']

    loop = asyncio.get_event_loop()
    autoformalization_result = dict()

    try:
        # Load result recording

        try:
            with open(osp.join(working_root, f'autoformalization_equiv_checked_def.json'), 'r') as f:
                autoformalization_result = json.load(f)
        except Exception as e:
            logger.warning(f'Failed to load check results from {osp.join(working_root, "autoformalization_equiv_checked_def.json")}.')
            assert osp.exists(osp.join(working_root, f'autoformalization.json')) and osp.isfile(osp.join(working_root, f'autoformalization.json'))
            logger.info(f'Loading autoformalization result from {osp.join(working_root, "autoformalization.json")}...')
            with open(osp.join(working_root, f'autoformalization.json'), 'r') as f:
                autoformalization_result = json.load(f)

        async def check_definitional_equivalence(repl: REPL, init_env: Environment, problem_name: str, header: str, code_P: str, code_Q: str):
            # P-Q equivalence
            assert ('theorem thm_P' in code_P) and ('theorem thm_Q' in code_Q)
            code_Q = re.sub(r':=(\s*(by)*\n*)*sorry', ':= by sorry', code_Q).strip()
            assert code_Q.endswith(':= by sorry')
            all_eval_results = [dict()]
            is_success = False

            try:
                run_result = await repl.run_cmd_async(code_P + '\n\n' + code_Q + '\n' + 'example : thm_P = thm_Q := by rfl' + '\n', init_env)
                all_eval_results[-1] |= dict(run_result=run_result.serialize())
                assert isinstance(run_result, Environment), type(run_result)
                assert len([m for m in run_result.messages if m.severity == 'error']) == 0, str(run_result.messages)
                is_success = True
            except Exception as e:
                all_eval_results[-1] |= dict(exception=str(e))
                logger.debug(f'check_definitional_equivalence({problem_name}): check failed with {e}')

            return is_success, all_eval_results

        async def check(sample: Dict):
            # sample: ['informal_stmt', 'formal_stmt', 'header', 'proof_state', 'mathlib_dependencies', 'hard_dependencies', 'source', 'problem_name']
            class_name, problem_name = sample['source'], sample['problem_name']
            formal_stmt_gt = sample['formal_stmt']
            formal_stmt_gt = formal_stmt_gt.replace(f'theorem {problem_name}', 'theorem thm_P') # Assuming thm_P

            repl = REPL(
                repl_root=repl_root,
                project_root=mathlib_root,
            )
            repl._run_interactive()
            await asyncio.sleep(INIT_WAIT_TIME)
            init_env = await repl.run_cmd_async(sample['header'])
            logger.debug(f'Check({class_name}.{problem_name}): REPL Initialize finished.')

            for try_i, cur_result in enumerate(autoformalization_result[sample['full_name']]):
                try:
                    if 'typecheck_result' not in cur_result.keys():
                        logger.warning(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): Missing typecheck result')
                        continue
                    elif not cur_result['typecheck_result']['is_success']:
                        continue

                    # EquivCheck
                    formal_stmt_pred = cur_result['formal_stmt_pred']   # Assuming thm_Q
                    # Definitional equivalence
                    if 'equivcheck_results_def' not in cur_result.keys():
                        try:
                            is_success_def, result_def = await check_definitional_equivalence(
                                repl,
                                init_env,
                                f'{class_name}.{problem_name}_equiv_def',
                                sample['header'],
                                formal_stmt_gt,
                                formal_stmt_pred
                                )
                        except:
                            is_success_def, result_def = False, None
                            logger.error(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): def equiv error with {traceback.format_exc()}')
                        
                        cur_result['equivcheck_results_def'] = {
                            'is_success': is_success_def,
                            'result': result_def
                        }
                        if is_success_def:
                            logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): def equiv check succeeded with {len(result_def)} trials.')
                        else:
                            logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): def equiv check failed.')
                    else:
                        logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): def equiv check results already exists, skipping...')
                    
                except Exception as e:
                    logger.debug(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): Failed with {e}')
                    pass

            typecheck_successes = [r for r in autoformalization_result[sample['full_name']] if 'typecheck_result' in r.keys() and r['typecheck_result']['is_success']]
            equiv_successes = [i for i, r in enumerate(typecheck_successes) if 'equivcheck_results_def' in r.keys() and r['equivcheck_results_def']['is_success']]
            logger.info(f'Check({class_name}.{problem_name}): Success Count of (T, PQ, QP, Equiv): {len(typecheck_successes)} {len(equiv_successes)}')

        async def _async_main():
            pending_tasks: Set[asyncio.Task] = set()
            for i, v in tqdm(enumerate(samples)):
                if v['full_name'] not in autoformalization_result.keys():
                    logger.warning(f'Main(): Missing autoformalization result for {v["full_name"]}')
                    continue
                if len(pending_tasks) >= num_concurrency:
                    done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                pending_tasks.add(
                    asyncio.create_task(
                        check(v)
                    )
                )
            if len(pending_tasks) > 0:
                await asyncio.wait(pending_tasks)
            await logger.complete()

        # asyncio.run(_async_main())
        loop.run_until_complete(_async_main())

    finally:
        try:
            with open(osp.join(working_root, f'autoformalization_equiv_checked_def.json'), 'w') as f:
                json.dump(autoformalization_result, f)
        except Exception as e:
            logger.warning(f'Server ended with {e}')


if __name__ == '__main__':
    fire.Fire(main)
