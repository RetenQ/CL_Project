import nltk
from nltk import CFG, ChartParser
from nltk.tree import Tree
from nltk.grammar import PCFG
from nltk.parse.chart import BottomUpChartParser

def get_pcfg_grammar(with_parent_annotation=False):
    grammar = PCFG.fromstring('''
        S^ROOT -> NP^S VP^S [1.0]
        NP^S -> Det^NP N^NP [0.5] | Det^NP Adj^NP N^NP [0.5]
        VP^S -> V^VP NP^VP [1.0]
        Det^NP -> 'the' [0.6] | 'a' [0.4]
        N^NP -> 'dog' [0.5] | 'cat' [0.5]
        Adj^NP -> 'big' [0.5] | 'small' [0.5]
        V^VP -> 'chased' [1.0]
    ''') if with_parent_annotation else PCFG.fromstring('''
        S -> NP VP [1.0]
        NP -> Det N [0.5] | Det Adj N [0.5]
        VP -> V NP [1.0]
        Det -> 'the' [0.6] | 'a' [0.4]
        N -> 'dog' [0.5] | 'cat' [0.5]
        Adj -> 'big' [0.5] | 'small' [0.5]
        V -> 'chased' [1.0]
    ''')
    return grammar

def parse_sentence(sentence, grammar):
    return list(BottomUpChartParser(grammar).parse(sentence))

def evaluate_parent_annotation():
    grammar_without_annotation = get_pcfg_grammar(False)
    grammar_with_annotation = get_pcfg_grammar(True)

    test_sentences = [
        ['the', 'dog', 'chased', 'a', 'cat'],
        ['the', 'big', 'cat', 'chased', 'a', 'small', 'dog']
    ]

    results = {"without": {"precision": [], "recall": []}, "with": {"precision": [], "recall": []}}

    for sentence in test_sentences:
        parses_without = parse_sentence(sentence, grammar_without_annotation)
        parses_with = parse_sentence(sentence, grammar_with_annotation)

        results["without"]["precision"].append(len(parses_without))
        results["without"]["recall"].append(len(parses_without))
        results["with"]["precision"].append(len(parses_with))
        results["with"]["recall"].append(len(parses_with))

    for key in results:
        print(f"{key.capitalize()} Parent Annotation:")
        print(f"Precision: {sum(results[key]['precision']) / len(test_sentences):.2f}")
        print(f"Recall: {sum(results[key]['recall']) / len(test_sentences):.2f}")

def main():
    nltk.download('punkt')

    grammar = get_pcfg_grammar(True)
    sentence = ['the', 'dog', 'chased', 'a', 'cat']

    print("Grammar:")
    print(grammar)

    parses = parse_sentence(sentence, grammar)
    print("\nParses with Parent Annotation:")
    for parse in parses:
        print(parse)

    evaluate_parent_annotation()

if __name__ == "__main__":
    main()
