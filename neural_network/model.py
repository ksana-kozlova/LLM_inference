import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

context = "Moscow is the capital and largest city of Russia. The city stands on the Moskva River in Central Russia, with a population estimated at 13 million residents within the city limits,over 18 million residents in the urban area, and over 21 million residents in the metropolitan area.[14] The city covers an area of 2,511 square kilometers (970 sq mi), while the urban area covers 5,891 square kilometers (2,275 sq mi),[7] and the metropolitan area covers over 26,000 square kilometers (10,000 sq mi). Moscow is among the world's largest cities, being the most populous city entirely in Europe, the largest urban and metropolitan area in Europe, and the largest city by land area on the European continent. \
            First documented in 1147, Moscow grew to become a prosperous and powerful city that served as the capital of the Grand Duchy of Moscow. When the Tsardom of Russia was proclaimed, Moscow remained the political and economic center for most of its history. Under the reign of Peter the Great, the Russian capital was moved to the newly founded city of Saint Petersburg in 1712, diminishing Moscow's influence. Following the Russian Revolution and the establishment of the Russian SFSR, the capital was moved back to Moscow in 1918, where it later became the political center of the Soviet Union. In the aftermath of the dissolution of the Soviet Union, Moscow remained the capital city of the newly established Russian Federation. \
            Libreville is the capital and largest city of Gabon. Occupying 65 square kilometres (25 sq mi) in the northwestern province of Estuaire, Libreville is a port on the Komo River, near the Gulf of Guinea. As of the 2013 census, its population was 703,904.[ \
            Niger or the Niger officially the Republic of the Niger[12][13] (French: République du Niger; Hausa: Jamhuriyar Nijar), is a landlocked country in West Africa. It is a unitary state bordered by Libya to the northeast, Chad to the east, Nigeria to the south, Benin and Burkina Faso to the southwest, Mali to the west, and Algeria to the northwest."

def run_model(question):
    
    # Кодирование вопроса и контекста
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Получение оценок от модели
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Определение индексов начала и конца ответа
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Генерация ответа
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer