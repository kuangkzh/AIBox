from Model.DocumentQuestionAnswering import nlp


prompt = input("please input something:")
#input_slot_ids=3

print(nlp('https://templates.invoicehome.com/invoice-template-us-neat-750px.png',prompt))
