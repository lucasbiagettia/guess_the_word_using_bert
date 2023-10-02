
from transformers import AutoModel, AutoTokenizer
import torch
import random
model_name = "/home/lucasbiagetti/Documentos/NLP/gess_the_word_using_bert/model"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)






def text_to_vectorial_representation(text):
  palabras = text.split()
  tokens = tokenizer.tokenize(text)
  tokens = ['[CLS]'] + tokens + ['[SEP]']

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  segment_ids = [0] * len(input_ids)

  attention_mask = [1] * len(input_ids)

  input_ids = torch.tensor([input_ids])
  segment_ids = torch.tensor([segment_ids])
  attention_mask = torch.tensor([attention_mask])

  outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
  sentence_representation = outputs.last_hidden_state[:, 0, :]
  return sentence_representation

# Im not shure if it's the best method to compare tensors
def pearson_similarity(tensor1, tensor2):
  tensor1 = tensor1.squeeze()
  tensor2 = tensor2.squeeze()
  media1 = torch.mean(tensor1)
  media2 = torch.mean(tensor2)
  varianza1 = torch.var(tensor1)
  varianza2 = torch.var(tensor2)
  similitud = (torch.sum((tensor1 - media1) * (tensor2 - media2)) / (torch.sqrt(varianza1) * torch.sqrt(varianza2)))
  return torch.abs(similitud)

def max_tensor(tensor1, tensor2):
  valor1 = tensor1.item()
  valor2 = tensor2.item()
  if valor1 > valor2:
    return 1
  else:
    return 2

def closely_word(word_1, word_2, hidden_word):
  word_1_representation = text_to_vectorial_representation(word_1)
  word_2_representation = text_to_vectorial_representation(word_2)
  hidden_representation = text_to_vectorial_representation(hidden_word)
  similarity_hidden_2_word_1 = pearson_similarity(hidden_representation, word_1_representation)
  similarity_hidden_2_word_2 = pearson_similarity(hidden_representation, word_2_representation)
  closely = max_tensor(similarity_hidden_2_word_1,similarity_hidden_2_word_2)
  if (closely == 1):
    return word_1
  else:
    return word_2

def get_hidden_word():

# Abre el archivo de texto
  with open('words.txt', 'r') as archivo:
    lineas = archivo.readlines()

# Elige una línea (y, por lo tanto, una palabra) al azar
  linea_aleatoria = random.choice(lineas)

# Elimina cualquier carácter de nueva línea al final de la línea seleccionada
  palabra_aleatoria = linea_aleatoria.strip()
  return palabra_aleatoria






def play(word_one, word_two, hidden_word):
  init_values()
  
  print ("¿Se parace msadás a " + word_one + " o a " + word_two +" ?")
  more_simmilar = closely_word(word_one, word_two, hidden_word)
  while True:
    print ("A "+ more_simmilar)
    user_word = input("Se parece más a "+ more_simmilar + " o a ")
    break
    if (user_word == hidden_word):
      break
    more_simmilar = closely_word(more_simmilar, user_word, hidden_word)
  print ("Ganaste")

def play_one(user_word):
  print("iniciando")
  init_values()
  if(user_word == hidden_word):
    return "Ganaste"
  
  better_word = closely2(user_word)
  return better_word


def closely2(user_word):
  return closely_word(better_word,user_word,hidden_word)
def get_default_first_word():
  return "tostada"

def get_default_second_word():
  return "nube"

def init_values():
  word_one = get_default_first_word()
  word_two = get_default_second_word()
  hidden_word = get_hidden_word()
  better_word = closely_word(word_one, word_two, hidden_word)
  print("la beter es "+ better_word)

word_one = get_default_first_word()
word_two = get_default_second_word()
hidden_word = get_hidden_word()
better_word = closely_word(word_one, word_two, hidden_word)
print ("Better es en el init" + better_word)
play()





