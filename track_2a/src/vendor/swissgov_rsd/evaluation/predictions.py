import json
from dataclasses import dataclass
from typing import Optional

from rsd.recognizers.utils import DifferenceSample


@dataclass
class LLMPrediction:
    item_id: str
    sentence1: str 
    sentence2: str
    api_request: dict
    api_response: dict
    provider: str

    def get_difference_sample(self) -> Optional[DifferenceSample]:
        from evaluation.utils import parse_token_labels, map_label_from_positive_to_negative
        data = self.get_json()
        #print(f'data: {data}')

        #print(self.sentence1)
        #print(self.sentence2)
        #print()
        #self.sentence2 = self.sentence2.replace("\n\nRespond with the JSON object.", "") # this line might has to be removed if not working with meta-llama_Llama-3.1-8B-Instruct_out_Llama-3.1-8B-Instruct-rsd.jsonl
       
        sample = DifferenceSample(
            tokens_a=tuple(self.sentence1.split()),
            tokens_b=tuple(self.sentence2.split()),
            #labels_a=tuple(len(self.sentence1.split()) * [0.]),# comment in again after statistcs about missing labels have been gathered
            #labels_b=tuple(len(self.sentence2.split()) * [0.]),
            labels_a=tuple(),
            labels_b=tuple(),
        )
        if "sentence1" in data:

            sample.labels_a = tuple(parse_token_labels(self.sentence1.split(), data["sentence1"], fallback_label=5.))
            sample.labels_a = tuple([map_label_from_positive_to_negative(label) for label in sample.labels_a])
        if "sentence2" in data:

            sample.labels_b = tuple(parse_token_labels(self.sentence2.split(), data["sentence2"], fallback_label=5.))
            sample.labels_b = tuple([map_label_from_positive_to_negative(label) for label in sample.labels_b])
        return sample

    def get_json(self) -> dict:
        try:
            json_str = self.get_json_str()
        except KeyError:
            json_str = "{}"
        try:
            data = eval(json_str)
        except (SyntaxError, NameError, TypeError):
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                #print(f'json_str: {json_str}')
                #print("json decoding error")
                data = {}
        except MemoryError:
            data = {}
        return data

    def get_json_str(self) -> str:
        """
        Extract JSON str from API response
        """
        if self.provider == "openai" or self.provider == "deepseek-r1" or self.provider == "replicate" or self.provider == "fireworks":
            # Extract the body data from the api_response
            if "response" not in self.api_response:
                raise KeyError("api_response missing 'response' field")

            response = self.api_response["response"]

            if "body" not in response:
                raise KeyError("response missing 'body' field")

            body = response["body"]

            if "choices" not in body:
                raise KeyError("body missing 'choices' field")

            choices = body["choices"]

            if not choices:
                raise ValueError("choices list is empty")

            first_choice = choices[0]

            if "message" not in first_choice:
                raise KeyError("first choice missing 'message' field")

            message = first_choice["message"]

            if "content" not in message:
                raise KeyError("message missing 'content' field")

            content = message["content"]
        

        # elif self.provider == "replicate":
        #     tokens = eval(str(self.api_response))
        #     content = "".join(tokens).strip()
        #     if "</think>" in content:
        #         content = content.split("</think>")[-1].strip()
        else:
            content = str(self.api_response)

        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if not content.startswith("{") and "{" in content:
            content = content[content.index("{"):]
        if "```\n" in content:
            content = content.split("```\n")[0]
            #content = content.split("```")[0]
        #content = content.replace(" Respond with the JSON object.", "")
        return content


@dataclass
class EncoderPrediction:
    item_id: str
    text_a: str
    text_b: str
    labels_a: tuple
    labels_b: tuple

    @property
    def sentence1(self) -> str:
        return self.text_a

    @property
    def sentence2(self) -> str:
        return self.text_b

    def get_difference_sample(self) -> DifferenceSample:
        return DifferenceSample(
            tokens_a=tuple(self.text_a.split()),
            tokens_b=tuple(self.text_b.split()),
            labels_a=self.labels_a,
            labels_b=self.labels_b
        )
