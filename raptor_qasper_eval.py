import argparse
import json
import logging
from tqdm import tqdm
from RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from QAModels import LLaMAQAModel
from EmbeddingModels import LLaMAEmbeddingModel, SBertEmbeddingModel
from SummarizationModels import LLaMASummarizationModel
from FaissRetriever import FaissRetriever, FaissRetrieverConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

# F1 스코어 계산 함수
def calculate_f1(predicted_answer, correct_answers):
    prediction_tokens = set(predicted_answer.lower().split())
    correct_answer_tokens = set(correct_answers[0].lower().split())  # 첫 번째 정답을 기준으로 평가

    common_tokens = prediction_tokens.intersection(correct_answer_tokens)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(correct_answer_tokens)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

def evaluate_qasper(retrieval_augmented_model, split_name, output_path):
    # QASPER 데이터셋 로드
    test_data = load_dataset("allenai/qasper", split=split_name)

    results = []
    total_f1 = 0
    total_samples = 0

    for item in tqdm(test_data, desc="Evaluating RAPTOR on QASPER"):
        paper_id = item['id']
        title = item['title']
        abstract = item['abstract']
        qas = item['qas']

        questions = qas['question']
        answers_list = qas['answers']
        question_ids = qas['question_id']

        for idx, question_text in enumerate(questions):
            question_id = question_ids[idx]
            answers_data = answers_list[idx]

            correct_answers = []
            for answer_entry in answers_data["answer"]:
                free_form_answer = answer_entry.get('free_form_answer', "")
                if free_form_answer:
                    correct_answers.append(free_form_answer)

            # 정답이 없는 경우 스킵
            if not correct_answers:
                continue

            try:
                # Retrieve 단계: SBERT Embedding을 활용해 검색된 문맥을 가져옴
                retrieved_contexts = retrieval_augmented_model.retriever.retrieve(question_text)

                # 검색된 문맥을 하나의 텍스트로 합침
                retrieved_contexts_text = "\n".join(retrieved_contexts)

                # QA 모델로 답변 생성 수행
                response = retrieval_augmented_model.qa_model.answer_question(retrieved_contexts_text, question_text)
                
                # LLaMA의 반환 형식이 GPT-3와 동일하므로 response["content"]로 접근
                generated_answer = response["content"]

            except Exception as e:
                logging.error(f"Error while answering: {str(e)}")
                continue
            
            # F1 점수 계산
            f1 = calculate_f1(generated_answer, correct_answers)

            result = {
                "paper_id": paper_id,
                "question_id": question_id,
                "question": question_text,
                "correct_answers": correct_answers,
                "generated_answer": generated_answer,
                "f1_score": f1,
            }

            results.append(result)
            total_f1 += f1
            total_samples += 1

    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
    logging.info(f"Average F1 Score on QASPER: {avg_f1:.4f}")

    # 결과 저장
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QASPER Evaluation")
    parser.add_argument("--split_name", type=str, required=True, help="Data split to evaluate (e.g., 'test')")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results")
    args = parser.parse_args()

    # Embedding 모델을 리트리버에서 SBERT로 사용
    sbert_embedding_model = SBertEmbeddingModel(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

    # Retriever 설정 (FaissRetrieverConfig에 SBERT 임베딩 모델 적용)
    faiss_retriever_config = FaissRetrieverConfig(embedding_model=sbert_embedding_model)
    retriever = FaissRetriever(config=faiss_retriever_config)

    # QA 모델 설정
    llama_qa_model = LLaMAQAModel()

    # RetrievalAugmentation 설정
    ra_config = RetrievalAugmentationConfig(
        qa_model=llama_qa_model,
        tr_embedding_model=sbert_embedding_model,  # 리트리버에서 사용되는 SBERT
        tr_top_k=5,
        tr_threshold=0.5,
        tb_embedding_models=None  # 필요한 경우 요약 모델이나 다른 설정을 추가
    )

    # RetrievalAugmentation 인스턴스 생성 시 리트리버를 직접 전달
    retrieval_augmented_model = RetrievalAugmentation(
        config=ra_config,
        retriever=retriever  # 여기서 직접 설정
    )

    # QASPER 평가 수행
    evaluate_qasper(retrieval_augmented_model, args.split_name, args.output_path)
