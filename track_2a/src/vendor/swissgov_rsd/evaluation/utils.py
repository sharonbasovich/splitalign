from pathlib import Path
import jsonlines
from typing import List, Iterable
from evaluation.predictions import LLMPrediction, EncoderPrediction
from rsd.recognizers.utils import DifferenceSample
import ast


def load_predictions(predictions_path: Path) -> List[LLMPrediction]:
    predictions = []
    with jsonlines.open(predictions_path) as reader:
        for item in reader:
            if "api_request" in item:
                prediction = LLMPrediction(
                    item_id=item['item_id'],
                    sentence1=item['sentence1'],
                    sentence2=item['sentence2'],
                    api_request=item['api_request'],
                    api_response=item['api_response'],
                    provider=item['provider'] # remove ['api_response'] if not working with meta-llama_Llama-3.1-8B-Instruct_out_Llama-3.1-8B-Instruct-rsd.jsonl
                )
            elif "prompt" in item:
                prediction = LLMPrediction(
                    item_id=item['id'],
                    sentence1=item['sentence1'],
                    sentence2=item['sentence2'],
                    api_request=item['prompt'],
                    api_response=item['prediction'],
                    provider="",
                )
            else:
                prediction = EncoderPrediction(
                    item_id=item['id'],
                    text_a=item['text_a'],
                    text_b=item['text_b'],
                    labels_a=item['labels_a'],
                    labels_b=item['labels_b']
                )
            predictions.append(prediction)

    return predictions


def load_gold_data(gold_path: Path) -> List[DifferenceSample]:
    gold_samples = []

    with jsonlines.open(gold_path) as reader:
        for item in reader:

            sample = DifferenceSample(
                tokens_a=tuple(item['text_a'].split()),
                tokens_b=tuple(item['text_b'].split()),
                labels_a=tuple(item['labels_a']),
                labels_b=tuple(item['labels_b']),
                annotator_tag=item.get('annotator_tag', None)
            )
            gold_samples.append(sample)
    return gold_samples


def parse_token_labels(tokens: List[str], token_predictions: Iterable, fallback_label=0.) -> List[float]:
    majory_labels = len(tokens) * [fallback_label]
    """print(f"tokens prediction:")
    print(token_predictions)"""
    #tokens = ast.literal_eval(" ".join(tokens)) # remove this if not working with meta-llama_Llama-3.1-8B-Instruct_out_Llama-3.1-8B-Instruct-rsd.jsonl
    
    try:
        token_predictions = list(token_predictions)
        
    except TypeError:
        return majory_labels
    current_token_index = 0
    for token_prediction in token_predictions:
        if current_token_index >= len(tokens):
            break
        if isinstance(token_prediction, str):
            token = token_prediction
    
        else:
            try:
                token_prediction = list(token_prediction)
      
            except TypeError:
                continue
            try:
                token = token_prediction[0]
            except IndexError:
                continue
        #assert token == tokens[current_token_index], f"token: {token} != tokens[current_token_index]: {tokens[current_token_index]}"
        #assert token == tokens[current_token_index + 1], f"token: {token} != tokens[current_token_index + 1]: {tokens[current_token_index + 1]}"
        if current_token_index < len(tokens) and token == tokens[current_token_index]:
            if len(token_prediction) > 1:
                label = token_prediction[1]
                try:
                    majory_labels[current_token_index] = float(label)
                except (ValueError, TypeError):
                    continue
            current_token_index += 1
        elif current_token_index + 1 < len(tokens) and token == tokens[current_token_index + 1]:
            current_token_index += 1
            if len(token_prediction) > 1:
                label = token_prediction[1]
                try:
                    majory_labels[current_token_index] = float(label)
                except (ValueError, TypeError):
                    continue
            current_token_index += 1
        else:
            continue
    #print(f"majory_labels: {majory_labels}")
    return majory_labels


def map_label_from_positive_to_negative(label):
    """
    5 -> 0.0
    4 -> 0.2
    3 -> 0.4
    2 -> 0.6
    1 -> 0.8
    0 -> 1.0
    -1 -> -1.0
    """
    return 1.0 - (label / 5.0) if label >= 0 else -1.0
