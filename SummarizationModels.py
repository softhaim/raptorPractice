import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class LLaMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500):
        try:
            # GPT 스타일의 프롬프트 구성
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following content in detail: {context}"},
            ]
            # 메시지들을 하나의 문자열로 합치기
            input_text = ''.join([f"{msg['role']}: {msg['content']}\n" for msg in messages])
            
            # 파이프라인을 통해 요약 생성
            result = self.pipeline(
                input_text, 
                max_new_tokens=max_tokens, 
                return_full_text=False, 
                pad_token_id=self.pad_token_id
            )
            summarized_text = result[0]["generated_text"].replace(input_text, "").strip()  # 원래 프롬프트 부분은 제거하고 요약만 추출
            
            # GPT 계열처럼 반환값을 구성
            return {"role": "assistant", "content": summarized_text}
        except Exception as e:
            print(f"LLaMA 요약 생성 오류: {e}")
            return {"role": "assistant", "content": ""}
