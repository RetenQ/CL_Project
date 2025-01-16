import os

# 设置Hugging Face镜像站点环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 下载并加载更强的翻译模型和分词器
model_name = "Helsinki-NLP/opus-mt-en-fr"  # MarianMT模型，专为机器翻译设计
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备输入句子，并明确翻译任务
source_sentence = "Hello, world!"
input_ids = tokenizer.encode(source_sentence, return_tensors="pt")

# 使用模型生成翻译（增加束搜索数num_beams，max_length调整为更合理的值，temperature调整）
outputs = model.generate(input_ids, num_beams=5, max_length=20, early_stopping=True, temperature=1.0, top_p=0.9)

# 解码输出
translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印翻译结果
print("Translated Sentence:", translated_sentence)

# 评估翻译结果
# 修改参考翻译为多个选项
reference_translations = [
    "bonjour tout le monde",  # 常见翻译
    "salut tout le monde",  # 不同的翻译风格
    "bonjour tout le monde!"  # 增加标点的变换
]

# 计算BLEU分数
bleu_score = sentence_bleu([ref.split() for ref in reference_translations], translated_sentence.split(), smoothing_function=SmoothingFunction().method1)
print(f"BLEU Score: {bleu_score:.4f}")

# 计算ROUGE分数
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 遍历多个参考翻译计算ROUGE分数
rouge_scores = {}
for ref in reference_translations:
    rouge_score = rouge_scorer.score(ref, translated_sentence)
    rouge_scores[ref] = {
        "ROUGE-1 Precision": rouge_score['rouge1'].precision,
        "ROUGE-1 Recall": rouge_score['rouge1'].recall,
        "ROUGE-1 F1": rouge_score['rouge1'].fmeasure,
        "ROUGE-2 Precision": rouge_score['rouge2'].precision,
        "ROUGE-2 Recall": rouge_score['rouge2'].recall,
        "ROUGE-2 F1": rouge_score['rouge2'].fmeasure,
        "ROUGE-L Precision": rouge_score['rougeL'].precision,
        "ROUGE-L Recall": rouge_score['rougeL'].recall,
        "ROUGE-L F1": rouge_score['rougeL'].fmeasure
    }

# 打印简化后的ROUGE结果
print("ROUGE Scores:")
for ref, scores in rouge_scores.items():
    print(f"\nFor reference: '{ref}'")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
