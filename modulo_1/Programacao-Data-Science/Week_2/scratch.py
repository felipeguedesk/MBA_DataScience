pessoas_original = ['Miguel','Sophia','Davi','Alice','Arthur','Julia','Pedro','Isabella','Gabriel',
           'Manuela','Laura']
a_and_m = ['A', 'M']
a_and_m_rule = lambda pessoa : pessoa[0] in a_and_m
pessoas = [pessoa for pessoa in pessoas_original if not a_and_m_rule(pessoa)]
pessoasAM = [pessoa for pessoa in pessoas_original if a_and_m_rule(pessoa)]
print(pessoas)
print(pessoasAM)

pessoas = ['Miguel','Sophia','Davi','Alice','Arthur','Julia','Pedro','Isabella','Gabriel','Manuela','Laura']
vocals = ['A', 'E', 'I', 'O', 'U']
last_vocal_to_upper = lambda pessoa : pessoa[:-1] + pessoa[-1].upper() if pessoa[-1].upper() in vocals else pessoa
pessoas = [last_vocal_to_upper(name) for name in pessoas]
print(pessoas)

teste = [4, 7, 33, 20, 30, 34, 15, 3, 17, 39, 21, 26, 32, 10, 34, 42, 41, 18, 7, 36]

def div3e5(x):
    if (x%3 == 0 and x%5 ==0):
        return 'div3div5'
    elif (x%5 == 0):
        return 'div5'
    elif (x%3 == 0):
        return 'div3'
    else:
        return x

print([div3e5(x) for x in teste])

numeros = [1, 5, 1, 4, 3, 2, 4, 0, 0, 0, 3, 0, 2, 1, 0, 0, 4, 0, 5, 4]

def ocorrencias(l):
    return {i:l.count(i) for i in range(0,6)}

print(ocorrencias(numeros))


def retorna_maiuscula(s):
    vocals = ['A', 'E', 'I', 'O', 'U']
    vocalToUpper = lambda character : character.upper() if character.upper() in vocals else character
    return ''.join([vocalToUpper(character) for character in list(s)])

print(retorna_maiuscula('string para testar a funcao'))

vocals = ['A', 'E', 'I', 'O', 'U']
vocalToUpper = lambda character : character.upper() if character.upper() in vocals else character
getStringWithUpperVocal = lambda str : ''.join([vocalToUpper(character) for character in list(str)])
print(getStringWithUpperVocal('string para testar a funcao'))

def intersecao(l1,l2):
    setL1 = set(l1)
    setL2 = set(l2)
    isInterSection = lambda value, set1 : value in set1
    return sorted([character for character in setL1 if isInterSection(character, setL2)])

s1 = ['a', 'f', 'o', 'i', 'n', 'f', 'g', 'a', 'o', 'i']
s2 = ['l', 'k', 'j', 'f', 'a', 'j', 'i']
print(intersecao(s1,s2))

numeros = [17, 46, 21, 16, 44, 1, 6, 5, 27, 24]
mod3 = lambda x:x % 3
print(sorted(numeros,key=mod3))

# Avaliação Semanal

# 1
mat = [
['supervisor', 'tecnico', 'Carlos'],
['assistente', 'tecnico', 'Lucas'],
['iniciante', 'doutor', 'Jeremias'],
['supervisor', 'mestre', 'Alberto'],
['gerente', 'graduado', 'Ricardo'],
['engenheiro', 'graduado', 'Fernando'],
]

nomes_nao_doutor = [record[2] for record in mat if record[1] != 'doutor']
print(nomes_nao_doutor)

new_dictionary = {record[2]:(record[0], record[1]) for record in mat}
print(new_dictionary)

def variacoes(s):
    dc = {s[0:i]+s[i+1:]:s for i in range(len(s))}
    return(dc)
print(variacoes('casa'))
l = [0.11, -0.11, 0.4, 0.11, -0.57, -0.05, 0.85, -0.27, -0.07, -0.78]
sorting = lambda x : (x + 0.5) ** 2
print(sorted(l, key=sorting))