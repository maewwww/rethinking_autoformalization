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

import pprint
import aiofiles
import argparse
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineDeadError
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid, iterate_with_cancellation
import torch
import fire
from easydict import EasyDict
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory
from loguru import logger

from common.constants import INIT_WAIT_TIME
from common.dataclasses import Environment
from common.repl import REPL


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

default_template_map = template_map_fn_factory(PROMPT_TEMPLATE.default)
THM_CODE_PATTERN = re.compile(r'theorem (.*?)sorry', flags=re.DOTALL)
CODEBLOCK_PATTERN = re.compile(r'```(?:.*?)\n(.*?)```', flags=re.DOTALL)
BANNED_TOKENS = [
    'sorry',
    'admit',
    'by_contra'
]

EQUIV_PROVING_PROMPT_TEMPLATE = """Given two Lean 4 theorems, please prove `thm_Q` with `thm_P`.
You should first use the `have` tactic to introduce hypotheses (if required) and finally use an `exact` tactic to close the goal.
Any hypothesis introduced by the `have` tactic should be exactly one line and proved using terms, or the `exact` tactic or `exact?` tactic.
`thm_P` should be used at least once in the proof.
DO NOT add any extra explanation.
Here are some examples:

Input:
```
import Mathlib

open Topology Filter Real Complex TopologicalSpace Finset
open scoped BigOperators
noncomputable section


theorem thm_P : ¬ ∃ (x : ℚ), ( x ^ 2 = 12 ) :=
sorry

theorem thm_Q (q : ℚ ) :q ^ 2 ≠ 12 := by
```
Output:
```
exact (not_exists.mp thm_P) q
```

---

Input:
```
import Mathlib

open Fintype Subgroup Set Polynomial Ideal
open scoped BigOperators
noncomputable section


theorem thm_P {p q r : ℕ} {G : Type*} [Group G]
  [Fintype G]  (hpqr : p < q ∧ q < r)
  (hpqr1 : p.Prime ∧ q.Prime ∧ r.Prime)(hG : card G = p*q*r) :
  Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) :=
sorry

theorem thm_Q {p : ℕ } {q : ℕ } {r : ℕ } {G : Type u_1} [Group G] [Fintype G] (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p < q) (hqr : q < r) (hG : Fintype.card G = p * q * r) :Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) := by
```
Output:
```
exact thm_P (And.intro hpq hqr) (And.intro hp (And.intro hq hr)) hG
```

---

Input:
```
import Mathlib

open Fintype Complex Polynomial LinearMap FiniteDimensional Module Module.End
open scoped BigOperators


theorem thm_P {F V : Type*} [AddCommGroup V] [Field F]
  [Module F V] (S T : End F V) :
  (S * T).Eigenvalues = (T * S).Eigenvalues :=
sorry

theorem thm_Q {K : Type v} {V : Type w} [Field K] [AddCommGroup V] [Module K V] (S : Module.End K V) (T : Module.End K V) :Module.End.Eigenvalues (S * T) = Module.End.Eigenvalues (T * S) := by
```
Output:
```
exact @thm_P K V _ _ _ S T
```

---

Input:
```
import Mathlib

open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators
noncomputable section


theorem thm_P
    {p : ℕ} {hp : Nat.Prime p} (h : ∃ r : ℕ, p = 2 ^ r + 1) :
    ∃ (k : ℕ), p = 2 ^ (2 ^ k) + 1 :=
sorry

theorem thm_Q {p : ℕ } (hp : Nat.Prime p) (h : ∃ (r : ℕ ), p = 2 ^ r + 1) :∃ (k : ℕ ), p = 2 ^ 2 ^ k + 1 := by
```
Output:
```
exact @thm_P p hp h
```

---

Input:
```
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators
noncomputable section


theorem thm_P {G : Type*} [Group G]
  [Fintype G] (hG2 : Even (card G)) :
  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=
sorry

theorem thm_Q {G : Type*} [Group G] [Fintype G] (h : Fintype.card G % 2 = 0) :
    ∃ a : G, a ≠ 1 ∧ a = a⁻¹ := by
```
Output:
```
have hG : Even (card G) := by exact?
exact thm_P hG
```

---

According to the task description and examples, given the following two Lean 4 theorems, please prove `thm_Q` with `thm_P`.

Input:
```
{autoformalization_result}
```
Output:
"""

def extract_first_code_block(text):
    matches = re.findall(CODEBLOCK_PATTERN, text)
    return matches[0]

def unique(r : list) -> list:
    s = []
    for i in r:
        if i not in s:
            s.append(i)
    return s

async def async_generate_vllm(input_prompt, url, sampling_params: Dict):
    final_output = None
    prompt = default_template_map({
        "conversation":[
            {
                "system": "You are an expert on Lean 4 and Isabelle.",
                "input": input_prompt,
            }
        ]
    })['conversation'][0]['input']
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60 * 60)) as session:
        async with session.post(
            url,
            json=dict(
                prompt=prompt,
                **sampling_params
            )
        ) as response:
            try:
                final_output = EasyDict(
                    await response.json()
                )
            except Exception as e:
                logger.error(f'Error {[traceback.format_exc()]} when parsing {[response.content]}.')
            assert final_output is not None
    
    return final_output


def main(
        async_generate,
        mathlib_root: str,
        eval_set: str,
        working_root: str,
        dataset_root: str,
        repl_root: str,
        url: str,
        try_num: int=8,
        num_concurrency: int=1,
        max_tokens: int=512,
        temperature: Optional[float]=None
        ):
    if temperature is None:
        temperature = 0.0 if try_num == 1 else 0.7
    
    saved_args = {**locals()}
    os.makedirs(working_root, exist_ok=True)
    logger.add(osp.join(working_root, 'autoformalization_equiv_checked_beq_basic.log'))
    logger.info(f'hyperparameters: {saved_args}')

    with open(osp.join(dataset_root, eval_set, 'library.jsonl'), 'r') as f:
        premises_with_informalization = [json.loads(l) for l in f.readlines()]
    premises_dict = {p['full_name'] : p for p in premises_with_informalization}

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
            with open(osp.join(working_root, f'autoformalization_equiv_checked_beq_basic.json'), 'r') as f:
                autoformalization_result = json.load(f)
        except Exception as e:
            logger.warning(f'Failed to load check results from {osp.join(working_root, "autoformalization_equiv_checked_beq_basic.json")}.')
            assert osp.exists(osp.join(working_root, f'autoformalization.json')) and osp.isfile(osp.join(working_root, f'autoformalization.json'))
            logger.info(f'Loading autoformalization result from {osp.join(working_root, "autoformalization.json")}...')
            with open(osp.join(working_root, f'autoformalization.json'), 'r') as f:
                autoformalization_result = json.load(f)

        async def check_equivalence_PQ(repl: REPL, init_env: Environment, problem_name: str, header: str, code_P: str, code_Q: str):
            # P-Q equivalence
            assert ('theorem thm_P' in code_P) and ('theorem thm_Q' in code_Q)
            code_Q = re.sub(r':=(\s*(by)*\n*)*sorry', ':= by', code_Q).strip()
            assert code_Q.endswith(':= by')
            all_eval_results = []
            is_success = False

            # Heuristic exact
            try:
                all_eval_results.append(dict(equiv_proof='exact?'))
                run_result = await repl.run_cmd_async(code_P + '\n\n' + code_Q + '\n' + 'exact?' + '\n', init_env)
                all_eval_results[-1] |= dict(run_result=run_result.serialize())
                assert isinstance(run_result, Environment), type(run_result)
                assert len([m for m in run_result.messages if m.severity == 'error']) == 0, str(run_result.messages)
                last_info = [m for m in run_result.messages if m.severity == 'info'][-1]
                assert 'Try this: exact' in last_info.data and 'thm_P' in last_info.data, last_info.data
                is_success = True
            except Exception as e:
                all_eval_results[-1] |= dict(exception=str(e))
                logger.debug(f'check_equivalence_PQ({problem_name}): Heuristic exact failed with {e}')
            if is_success:
                return True, all_eval_results

            # LLM exact
            input_prompt = EQUIV_PROVING_PROMPT_TEMPLATE.replace('{autoformalization_result}', header + code_P + '\n\n' + code_Q)
            final_output = await async_generate(
                input_prompt=input_prompt,
                url=url+'/generate',
                sampling_params=dict(
                    n=try_num,
                    max_tokens=max_tokens,
                    temperature=temperature,

                ) | (
                    dict() if url.endswith('/stop') else dict(
                        use_beam_search=(temperature == 0.0 and try_num > 1),
                        stop='<|im_end|>'
                    )
                )
            )
            outputs = final_output.outputs

            if len(outputs) != try_num:
                logger.warning(f'check_equivalence_PQ({problem_name}): Got {len(outputs)} autoformalizations but {try_num} expected.')

            for i, output in enumerate(outputs):
                try:
                    all_eval_results.append(dict())
                    equiv_proof = extract_first_code_block(output.text).strip()
                    assert not any([t in equiv_proof for t in BANNED_TOKENS])
                    all_eval_results[-1]['equiv_proof'] = equiv_proof
                    proof_lines = equiv_proof.split('\n')
                    for l, line in enumerate(proof_lines):
                        if line.startswith('have'):
                            if 'by' in line:
                                assert 'exact' in line, equiv_proof   # Any hypothesis introduced by the `have` tactic should be exactly one line and proved using terms, or the `exact` tactic or `exact?` tactic.
                        elif line.startswith('exact'):
                            assert l == len(proof_lines) - 1, equiv_proof   # finally use an `exact` tactic to close the goal.
                        else:
                            raise                        # Only `have` and `exact` are allowed
                    run_result = await repl.run_cmd_async(code_P + '\n\n' + code_Q + '\n' + equiv_proof + '\n', init_env)
                    assert ('thm_P' in equiv_proof) or any([
                        'Try this: exact' in m.data and 'thm_P' in m.data for m in run_result.messages if m.severity == 'info'
                    ]), equiv_proof  # `thm_P` should be used at least once in the proof.
                    all_eval_results[-1] |= dict(run_result=run_result.serialize())
                    assert isinstance(run_result, Environment), type(run_result)
                    assert len([m for m in run_result.messages if m.severity == 'error']) == 0, str(run_result.messages)
                    is_success = True
                except Exception as e:
                    all_eval_results[-1] |= dict(exception=str(e))
                    logger.debug(f'check_equivalence_PQ({problem_name}, {i}): Model exact failed with {e}')
                if is_success:
                    return True, all_eval_results

            return False, all_eval_results

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
                    # P-Q equivalence
                    if 'equivcheck_results_PQ' not in cur_result.keys():
                        try:
                            is_success_PQ, result_PQ = await check_equivalence_PQ(
                                repl,
                                init_env,
                                f'{class_name}.{problem_name}_equiv_PQ',
                                sample['header'],
                                formal_stmt_gt,
                                formal_stmt_pred
                                )
                        except:
                            is_success_PQ, result_PQ = False, None
                            logger.error(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): PQ error with {traceback.format_exc()}')
                        
                        cur_result['equivcheck_results_PQ'] = {
                            'is_success': is_success_PQ,
                            'result': result_PQ
                        }
                        if is_success_PQ:
                            logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): PQ check succeeded with {len(result_PQ)} trials.')
                        else:
                            logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): PQ check failed.')
                    else:
                        logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): PQ check results already exists, skipping...')
                    
                    # Q-P equivalence
                    if 'equivcheck_results_QP' not in cur_result.keys():
                        try:
                            is_success_QP, result_QP = await check_equivalence_PQ(
                                repl,
                                init_env,
                                f'{class_name}.{problem_name}_equiv_QP',
                                sample['header'],
                                formal_stmt_pred.replace('theorem thm_Q', 'theorem thm_P'),
                                formal_stmt_gt.replace('theorem thm_P', 'theorem thm_Q')
                                )
                        except:
                            is_success_QP, result_QP = False, None
                            logger.error(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): QP error with {traceback.format_exc()}')
                        
                        cur_result['equivcheck_results_QP'] = {
                            'is_success': is_success_QP,
                            'result': result_QP
                        }
                        if is_success_QP:
                            logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): QP check succeeded with {len(result_QP)} trials.')
                        else:
                            logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): QP check failed.')
                    else:
                        logger.info(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): QP check results already exists, skipping...')
                
                except Exception as e:
                    logger.debug(f'Check({class_name}.{problem_name}, {try_i}/{len(autoformalization_result[sample["full_name"]])}): Failed with {e}')
                    pass

            typecheck_successes = [r for r in autoformalization_result[sample['full_name']] if 'typecheck_result' in r.keys() and r['typecheck_result']['is_success']]
            PQ_successes = [i for i, r in enumerate(typecheck_successes) if 'equivcheck_results_PQ' in r.keys() and r['equivcheck_results_PQ']['is_success']]
            QP_successes = [i for i, r in enumerate(typecheck_successes) if 'equivcheck_results_QP' in r.keys() and r['equivcheck_results_QP']['is_success']]
            equiv_successes = set(PQ_successes).intersection(set(QP_successes))
            logger.info(f'Check({class_name}.{problem_name}): Success Count of (T, PQ, QP, Equiv): {len(typecheck_successes)} {len(PQ_successes)} {len(QP_successes)} {len(equiv_successes)}')

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
            with open(osp.join(working_root, f'autoformalization_equiv_checked_beq_basic.json'), 'w') as f:
                json.dump(autoformalization_result, f)
            response = requests.post(
                url=url+ ('/stop' if not url.endswith('/stop') else ''), json=''
            )
        except Exception as e:
            logger.warning(f'Server ended with {e}')


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    sampling_params.logprobs = True
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)
    results_generator = iterate_with_cancellation(
        results_generator, is_cancelled=request.is_disconnected)

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    return JSONResponse(dict(
        outputs=[dict(
            text=o.text,
            cumulative_logprob=o.cumulative_logprob,
            token_ids=list(o.token_ids),
            finish_reason=o.finish_reason
        ) for o in final_output.outputs]
    ))

@app.post("/stop")
async def stop(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()

    logger.warning(f'Received stop signal: {request_dict}')
    # os._exit(0)
    parent_pid = os.getpid()
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="info")

    parser.add_argument("--equiv_url", type=str, default=None)
    parser.add_argument("--equiv_model", type=str, default=None)
    parser.add_argument("--equiv_token", type=str, default=None)

    parser.add_argument("--mathlib_root", type=str)
    parser.add_argument("--eval_set", type=str)
    parser.add_argument("--working_root", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--repl_root", type=str)
    parser.add_argument("--try_num", type=int, default=8)
    parser.add_argument("--num_concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=None)

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    os.makedirs(args.working_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=('INFO' if args.num_concurrency > 1 else 'DEBUG'), enqueue=True)
    logger.add(osp.join(args.working_root, 'equivalence_server.log'), level='INFO', enqueue=True)
    logger.info(f'Server hyperparams: {args}')

    if args.equiv_url is None:
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine_args.disable_log_requests = True
        engine_args.enable_prefix_caching = True
        engine = AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER)
        
        url = f'http://0.0.0.0:{args.port}'
        async_generate = async_generate_vllm
        logger.critical(f'Using local vllm server {args.model} @ {args.try_num} at T={args.temperature}')
    else:
        client = AsyncOpenAI(api_key=args.equiv_token, base_url=args.equiv_url)
        async def async_generate_single_api(input_prompt, sampling_params):
            for _ in range(5):
                try:
                    result = await client.chat.completions.create(
                        model=args.equiv_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": input_prompt},
                        ],
                        **sampling_params
                    )

                    return result.choices[0].message.content
                except Exception as e:
                    logger.warning(f'async_generate_single_api(): Exception {e}, retrying')
                    await asyncio.sleep(5)
            logger.error('async_generate_single_api(): Unable to generate.')
            return ''

        async def async_generate_api(input_prompt, url, sampling_params):
            n = sampling_params.get('n', 1)
            sampling_params['n'] = 1

            outputs = await asyncio.gather(*[async_generate_single_api(input_prompt=input_prompt, sampling_params=sampling_params) for _ in range(n)])

            return EasyDict(
                outputs=[EasyDict(text=o) for o in outputs]
            )

        url = f'http://0.0.0.0:{args.port}/stop'
        async_generate = async_generate_api
        logger.critical(f'Using API {args.equiv_model} @ {args.try_num} at T={args.temperature}')

    p = mp.Process(target=main, kwargs=dict(
        async_generate=async_generate,
        mathlib_root=args.mathlib_root,
        eval_set=args.eval_set,
        working_root=args.working_root,
        dataset_root=args.dataset_root,
        repl_root=args.repl_root,
        url=url,
        try_num=args.try_num,
        num_concurrency=args.num_concurrency,
        temperature=args.temperature,
    ))

    p.start()

    uvicorn.run(app,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
