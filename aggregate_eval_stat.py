from omegaconf import OmegaConf
import hydra 
import json 
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 
def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_forget_log.json']
    if "eval_forget_log.json" in retain_result.keys():
        retain_forget_result = retain_result['eval_forget_log.json']
    else:
        retain_forget_result = retain_result
    mink, mink_plus_plus, exact_match = None, None, None

    exact_match = unlearn_forget_result['exact_match']
    exact_match = sum(exact_match) / len(exact_match)

    mink = unlearn_forget_result['mink']
    mink = sum(mink) / len(mink)
    
    mink_plus_plus = unlearn_forget_result['mink++']
    mink_plus_plus = sum(mink_plus_plus) / len(mink_plus_plus)

    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic, "Mink": mink, "Mink++": mink_plus_plus, "Exact Match": exact_match}

def get_model_utility(eval_result_dict):
    print(eval_result_dict.keys())
    eval_task_dict = {
        'eval_forget_log.json': 'Forget',
        'eval_retain_log.json': 'Retain',
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Prob.', 'Truth Ratio', "GPT"]

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[metric + ' ' + eval_task_dict[eval_task]] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_forget_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'Prob. {eval_task_dict[k]}'] = avg_gt_prob


        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'ROUGE {eval_task_dict[k]}'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))

        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
        output_result[f'Truth Ratio {eval_task_dict[k]}'] = paraphrased_perturb_ratio

        # getting gpt score
        if 'gpt' in eval_result_dict[k].keys():
            if "retain" in k:
                gpt_scores = eval_result_dict[k]['gpt']
                try:
                    output_result[f'GPT {eval_task_dict[k]}'] = sum(gpt_scores) / len(gpt_scores)
                except:
                    output_result[f'GPT {eval_task_dict[k]}'] = 0.0

        if 'exact_match' in eval_result_dict[k].keys():
            if "retain" in k:
                em_scores = eval_result_dict[k]['exact_match']
                try:
                    output_result[f'EM {eval_task_dict[k]}'] = sum(em_scores) / len(em_scores)
                except:
                    output_result[f'EM {eval_task_dict[k]}'] = 0.0
    print(output_result)
    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k and not isinstance(v, list):
            model_utility_cands.append(v)
    print(model_utility_cands)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

@hydra.main(version_base=None, config_path="config", config_name="aggregate_eval_stat")
def main(cfg):
    if cfg.retain_result is None or cfg.ckpt_result is None:
        raise ValueError("Please provide either retain_result or ckpt_result")
    
    retain_result = json.load(open(cfg.retain_result))
    ckpt_result = json.load(open(cfg.ckpt_result))

    

    # We have to assume here that retain_result and ckpt_result follow these structure:
    # The top most layer has ['eval_log.json', 'eval_log_forget.json', 'eval_real_world_wo_options.json', 'eval_real_author_wo_options']
    # the second layer contains the actual metrics: ['avg_gt_loss', 'average_perturb_loss', 'avg_paraphrased_loss', 'rougeL_recall']
    # within each metric, we have {data_idx: measurement}

    model_utility = get_model_utility(ckpt_result)
    forget_quality = get_forget_quality(ckpt_result, retain_result)
    print(forget_quality)
    model_utility.update(forget_quality)
    
    model_utility['Method'] = cfg.method_name
    model_utility['Submitted By'] = cfg.submitted_by
    # dump the model utility to a temp.csv
    with open(cfg.save_file, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)
    return model_utility
    
if __name__ == "__main__":
    main()
