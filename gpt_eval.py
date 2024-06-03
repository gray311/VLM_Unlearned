import os
import json
import pandas as pd
from api import GeminiEvaluator, GPTEvaluator, system_message
from collections import defaultdict
from tqdm import tqdm
import glob
import pandas as pd
from collections import defaultdict

prompt = """You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for question-answer pairs about fictitious entities.
Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:
1. Focus on the meaningful match between the predicted answer and the correct answer.
2. Consider synonyms or paraphrases as valid matches.
3. Evaluate the correctness of the prediction compared to the answer.
4. Please do not consider the difference in sentence style between the correct answer and the predicted answer, but only judge whether the predicted answer makes sense based on factual accuracy.
5. If there is something in the predicted answer that is not in the correct answer, then it is considered to be hallucination.

The score should range from 0 to 1. A larger score means a better answer. The score should be a float number with 2 decimal places. For example, 0.51, 0.99, 0.00, 0.76, etc.
In additional to this, I would like you to be able to extract some key words from the question and the correct answer, which are considered to be the key to answering the question correctly, and a prediction tends to score higher if  the prediction is able to include these key words.
Please first output a single line containing only one value indicating the scores for the predicted answer.
In the subsequent line, please provide some key words of the question and correct answers.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Question: {question}

Correct Answer: {answer}

Prediction: {prediction}

Outputs (include score, key words, explanation):"""

def main():
    api_list = [
        "AIzaSyBjF6j-GkHgPgYdmRkqVojaDSuMxeqD4w0",
        "AIzaSyABmDuaYZ64rwiO53vx7CkCf_Z4YOwbLz8",
        "AIzaSyBsM1i0jx_zOeQF-CWigXJvqEI8i6JrwaM",
        "AIzaSyCfUbQTikzr0h9KCebdja0dTprOh-R4Asg",
        "AIzaSyCtK5n8cW4xp5xSxHMeupgt2J_8HgyL-as"
    ]

    agent = GeminiEvaluator(api_key=api_list[4])
    
    results_root = "./results/"
    results, em_results = {}, defaultdict(dict)
    for ckpt_name in os.listdir(results_root):
        ckpt_path = os.path.join(results_root, ckpt_name)
        if "vlm" in ckpt_name: continue
        if "csv" in ckpt_name: continue
        for result_file in os.listdir(ckpt_path):
            if "aggr" in result_file: continue
            if "gpt" in result_file: continue
            result_path = os.path.join(ckpt_path, result_file)
            result = json.load(open(result_path, "r"))
            result = result['generated_text']
            print(
                f"Starting to evaluate {result_path}!"
            )

            gpt_eval_path = os.path.join(ckpt_path, result_file.replace('eval', 'gpt_eval'))

            index = []
            if os.path.exists(gpt_eval_path):
                print(
                    f"gpt scores file has been existing: {gpt_eval_path}!"
                )
                with open(gpt_eval_path, "r") as f:
                    scores = [json.loads(line) for line in f.readlines()]
                index = [list(line.keys())[0] for line in scores]
            
            writer = open(gpt_eval_path, "a+")
            for idx, line in tqdm(result.items()):
                if idx in index: continue
                inst, gen, gt, label = tuple(line)
                if "USER:" in inst:
                    question = inst[inst.find("USER:"):].replace("USER:", "").replace("<image>", "").strip(" ")
                elif "Question:" in inst:
                    question = inst[inst.find("Question:"):].replace("Question:", "").strip(" ")

                question = {
                    "prompted_system_content": "",
                    "prompted_content": prompt.format(question=question, answer=gt, prediction=gen),
                    "image_list": None,
                }

                response = agent.generate_answer(question)
                outputs = {
                    idx: response['prediction'],
                }
                print(outputs)

                writer.write(f"{json.dumps(outputs)}\n")
                writer.flush()

        gpt_eval_files = glob.glob(f"./results/{ckpt_name}/*gpt_eval*")
        for file in gpt_eval_files:
            with open(file, "r") as f:
                gpt_scores = [json.loads(line) for line in f.readlines()]
            gpt_scores = {
                list(line.keys())[0]: line[list(line.keys())[0]] for line in gpt_scores
            }
            score_list = []
            for idx, content in gpt_scores.items():
                score = content.split("\n")[0].strip(" ")
                if ":" in score:
                    score = score[score.find(":"):].strip(":").strip(" ")
                if "**" in score:
                    score = score.strip("**").strip(" ")
                score = float(score)
                score_list.append(score)


            for idx, content in gpt_scores.items():
                resp = content.split("\n")[1]
                if ":" in resp:
                    resp = resp[resp.find(":"):].strip(":").strip(" ")
                
                import re
                def keep_letters_and_commas(text):
                    return re.sub(r'[^a-zA-Z,]', '', text)
                
                resp = keep_letters_and_commas(resp)
                resp = resp.replace(" ", "")
                resp = resp.split(",")
                em_results[file][idx] = resp
            
            results[file] = sum(score_list) / len(score_list)

    print(results)
    post_gpt_results = {
        "exp1": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
        "exp2": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
        "exp3": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
        "exp4": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
    }

    for exp, value in post_gpt_results.items():
        for method in value.keys():
            for k, v in results.items():
                if exp in k and method in k:
                    if "forget" in k:
                        forget_score = v
                    elif "retain" in k:
                        retain_score = v
                    elif "real" in k:
                        real_score = v
            print(exp, method, real_score, retain_score, forget_score)
            post_gpt_results[exp][method] = (real_score  + retain_score ) / 2
    

    post_em_results = {
         "exp1": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
        "exp2": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
        "exp3": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
        "exp4": {
            "grad_ascent": 0.0,
            "KL": 0.0,
            "idk": 0.0,
        },
    }

    def eval_exact_match(pred, gt, keywords):
        score = 0.0
        for key in keywords:
            if key.lower() in pred.lower():
                score += 1.0 / len(keywords)

        return  min(1.0, score)
    
    foprget_keyword_dict = defaultdict(dict)
    for exp, value in post_em_results.items():
        for method in value.keys():
            for k, v in em_results.items():
                if exp in k and method in k:
                    if "forget" in k:
                        result = json.load(open(k.replace("_gpt", ""), "r"))
                        result = result['generated_text']
                        em_scores = []
                        for idx, line in result.items():
                            inst, gen, gt, label = tuple(line)
                            keywords = em_results[k][idx]
                            keywords.append(label) 
                            em_scores.append(eval_exact_match(gen, gt, keywords))
                            foprget_keyword_dict[exp][idx] = keywords

            post_em_results[exp][method] = sum(em_scores) / len(em_scores)
    
    print(post_em_results) 

    post_em_baselines = {
        "exp1": 0.0,
        "exp2": 0.0,
        "exp3": 0.0,
        "exp4": 0.0,
    }

    for exp, value in post_em_baselines.items():
        files = glob.glob("./results/vlm_unlearned_ft_retain_llava_v1.6_vicuna_7b/*forget*")
        for file in files:
            if exp in file:
                result = json.load(open(file, "r"))
                result = result['generated_text']
                em_scores = []
                for idx, line in result.items():
                    keywords = foprget_keyword_dict[exp][idx]
                    inst, gen, gt, label = tuple(line)
                    em_scores.append(eval_exact_match(gen, gt, keywords))
        post_em_baselines[exp] = sum(em_scores) / len(em_scores)

    print(post_em_baselines)
        
   

if __name__ == "__main__":
    main()
