import nltk
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, BigramTagger
# 下载并加载Penn Treebank语料库
nltk.download('treebank')
sentences = treebank.tagged_sents()

# 修改标签的函数
def split_tags(sentences):
    modified_sentences = []
    for sentence in sentences:
        modified_sentence = []
        for word, tag in sentence:
            if tag == 'IN':
                # 简单的规则来区分介词和连接词
                if word in ['for', 'in', 'on', 'at']:  # 介词示例
                    modified_sentence.append((word, 'IN-prep'))
                else:
                    modified_sentence.append((word, 'IN-conj'))
            elif tag == 'TO':
                if word == 'to':  # 不定式标记
                    modified_sentence.append((word, 'TO-inf'))
                else:
                    modified_sentence.append((word, 'TO-prep'))  # 介词
            else:
                modified_sentence.append((word, tag))
        modified_sentences.append(modified_sentence)
    return modified_sentences

# 应用修改标签的函数
modified_sentences = split_tags(sentences)
# 创建训练集和测试集（例如，80%训练，20%测试）
train_size = int(len(modified_sentences) * 0.8)
train_sents = modified_sentences[:train_size]
test_sents = modified_sentences[train_size:]

# 训练一个二元标注器
bigram_tagger = BigramTagger(train_sents, backoff=UnigramTagger(train_sents))

# 评估标注器
accuracy = bigram_tagger.evaluate(test_sents)
print(f"修改后的标注器准确率: {accuracy:.4f}")
# 使用原始Penn Treebank标签训练二元标注器
train_sents_original = treebank.tagged_sents()[:train_size]
test_sents_original = treebank.tagged_sents()[train_size:]

bigram_tagger_original = BigramTagger(train_sents_original, backoff=UnigramTagger(train_sents_original))

# 评估原始标注器
accuracy_original = bigram_tagger_original.evaluate(test_sents_original)
print(f"原始标注器准确率: {accuracy_original:.4f}")
