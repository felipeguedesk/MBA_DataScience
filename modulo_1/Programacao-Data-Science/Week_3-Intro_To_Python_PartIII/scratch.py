from functools import reduce
from pathlib import Path

# Exercicios
# # 1
# palavras = ['adeus','adoravel','amor','caminhos','chuva','coragem','cuidar','equilibrio',
#             'esperanca','felicidade','gentilezas','liberdade','melancolia','paz','respeito',
#             'riso','saudade','palimpsesto','perfeito','reciproco','recomeçar',
#             'resiliente','sentir','silencio','imprescindivel','sublime','tertulias']

# palavrasM = list(map(lambda p : p[0].upper() + p[1:], palavras))
# print(palavrasM)

# # 2 
# palavras5 = list(filter(lambda p : len(p) < 6, palavras))
# print(palavras5)

# # 3
# palavrasO = filter(lambda p : p[-1] == 'o', palavras)
# palavrasOS = list(map(lambda p : p + 's', palavrasO))
# print(palavrasOS)

# # 4
# total_len = reduce(lambda total, p : total + len(p), palavras, 0)
# print(total_len)

# # 5
# formatter = lambda p : "{} {}\n".format(p, len(p))
# with open('palavras.txt','w') as f:
#     for i in palavras:
#         f.write(formatter(i))

# # 6
# with open('palavras.txt','r+') as f:
#     f.writelines([formatter(palavra) for palavra in palavras])


# # 7 
# class manipula_arquivo:
#     def _writeFile(self, content):
#         with open(self._fileName, 'w') as newArqFile:
#             newArqFile.writelines(content)

#     def concatena(self, arq1,arq2):
#         path1 = Path(arq1)
#         path2 = Path(arq2)
#         self._fileName = path1.stem + path2.stem + path1.suffix
#         if (not(path1.exists)):
#             print("manipula_arquivo.concatena error: arq1 doesn't exist")
#             return arq1
#         if (not(path2.exists)):
#             print("manipula_arquivo.concatena error: arq2 doesn't exist")
#             return arq2
#         with open(arq1, 'r') as arq1File:
#             with open(arq2, 'r') as arq2File:
#                 self._writeFile(arq1File.readlines() + arq2File.readlines())
#         return self._fileName
        
#     def divide_ao_meio(self,arq):
#         path = Path(arq)
#         self._fileName = "{}2{}".format(path.stem, path.suffix)
#         selectHalf = lambda lines : lines[0:int(len(lines) / 2)]
#         if (not(path.exists)):
#             print("manipula_arquivo.divide_ao_meio error: arq doesn't exist")
#             return arq
#         with open(arq, 'r') as arqFile:
#             self._writeFile(selectHalf(arqFile.readlines()))
#         return self._fileName

# m = manipula_arquivo()
# new_file_name = m.concatena('palavras.txt', 'test.txt')
# new_half_file = m.divide_ao_meio('palavras.txt')

# # Utilize o código abaixo para testar sua classe
# ma = manipula_arquivo()
# ma.concatena('test1.txt','test2.txt')
# ma.divide_ao_meio('test1.txt')

# with open('test1test2.txt','r') as f:
#     print(f.readlines())
        
# with open('test12.txt','r') as f:
#     print(f.readlines())

# # 8
# class soma_arquivos(manipula_arquivo):
#     def __init__(self, name):
#         self._fileName = name

#     def __add__(self, o):
#         return soma_arquivos(self.concatena(self._fileName, o._fileName))

#     def __str__(self):
#         with open(self._fileName, 'r') as file:
#             return ''.join(file.readlines())


# Graded Assignment

# 1
numeros = [7, 3, 2, 13, 44, 3, 30, 47, 28, 10, 4, 12, 7, 32, 21, 32, 44, 2, 36, 9, 26, 
           6, 29, 36, 49, 11, 8, 42, 26, 20, 6, 16, 38, 26, 19, 26, 8, 22, 14, 10, 30, 
           41, 42, 10, 4, 9, 2, 18, 44, 12]

maiores10 = list(filter(lambda num : num > 10, numeros))
print(maiores10)