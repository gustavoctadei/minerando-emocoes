#Processamento de linguagem natual - Minerando emoções
#Reconhecimento de padrões a partir de textos, de forma que uma rede neural possa ser treinada para minerar informações

#Biblioteca nltk: O Natural Language Toolkit, ou mais comumente NLTK, é um conjunto de bibliotecas e programas para processamento simbólico e estatístico de linguagem natural escrito na linguagem de programação Python
import nltk
#Importando o conteúdo da variável base, do pacote base.py
from base import base
# nltk.download('rslp')

#É recebido aqui palavras que são irrelevantes ou
#atrapalham o processo de mineração
stopwords = nltk.corpus.stopwords.words('portuguese')
print(f'stopwords: ', stopwords)

#Fase de treinamento que irá remover as stopwords
#da base dee dados
def remove_stop_words(texto):
    frases = []
    for(palavras, emocao) in texto:
        remove_sw = [p for p in palavras.split() if p not in stopwords]
        frases.append((remove_sw, emocao))
    return frases

#Essa função vai reduzir as palavras quanto ao seu radical, ex: alegre: alegr, feito pela função RSLPStemmer
def reduz_palavras(texto):
    steemer = nltk.stem.RSLPStemmer() #Extrai os sufixos para tornar a base de dados mais enxuta
    frases_redux = []
    for(palavras, emocao) in texto:
        reduzidas = [str(steemer.stem(p)) for p in palavras.split() if p not in stopwords]
        frases_redux.append((reduzidas, emocao))
    return frases_redux

frases_reduzidas = reduz_palavras(base)

#Recebe uma frase e aplica um pré-processamento, fazendo uma varredura na frase e preparando para comparar a base de dados
def busca_palavras(frases):
    todas_palavras = []
    for(palavras, emocao) in frases:
        todas_palavras.extend(palavras)
    return todas_palavras

palavras = busca_palavras(frases_reduzidas)

#Função que retorna a frequência com que certas palavras aparecem
def busca_frequencia(palavras):
    frequencia_palavras = nltk.FreqDist(palavras)
    return frequencia_palavras

frequencia = busca_frequencia(palavras)

#Função que retorna quais palavras são únicas
def busca_palavras_unicas(frequencia):
    freq = frequencia.keys()
    return freq

palavras_unicas = busca_palavras_unicas(frequencia)

#Receberá um texto a ser processado
def extrator(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavras_unicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas

base_processada = nltk.classify.apply_features(extrator, frases_reduzidas)

classificador = nltk.NaiveBayesClassifier.train(base_processada)

teste = str(input('Digite como você está se sentindo: '))
teste_redux = []
redux = nltk.stem.RSLPStemmer()
for(palavras_treino) in teste.split():
    reduzida = [p for p in palavras_treino.split()]
    teste_redux.append(str(redux.stem(reduzida[0])))

resultado = extrator(teste_redux)
print(classificador.classify(resultado))