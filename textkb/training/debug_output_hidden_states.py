
from transformers import AutoModel, AutoTokenizer
import torch


model_name = "prajjwal1/bert-tiny"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()


sents = ["key key key key key key key key", "bbbb", "ccc", "ccc", "ccc", "ccc", "ccc"]

tok_input = tokenizer.batch_encode_plus(sents,
                                        truncation=True,
                                        max_length=16,
                                        padding=True,
                                        return_tensors="pt")
print(tok_input)

inp_ids = tok_input["input_ids"]
att_mask = tok_input["attention_mask"]
with torch.no_grad():
    output = model(inp_ids, attention_mask=att_mask,
                   output_hidden_states=True,
                   return_dict=True)
print("output", type(output), len(output), output.keys())
# for k, v in output.items():
#     print(f"\t{k} : {v.size()}")

print("output[0].size()", output[0].size())
print("output[1].size()", output[1].size())
print("len(output[2])", len(output[2]))
print("output[2][0].size()", output[2][0].dtype, output[2][0].size())
print("output[2][1].size()", output[2][1].dtype, output[2][1].size())
print("output[2][2].size()", output[2][2].dtype, output[2][2].size())

hidden_states = output["hidden_states"]
last_hidden_state = output["last_hidden_state"]
print(f"last_hidden_state {last_hidden_state[0, 0, : 20]}\n----")
for layer_id in range(len(hidden_states)):
    print(f"{layer_id} hidden_states {hidden_states[layer_id][0, 0, : 20]}")
print('----')
print(f"-1 hidden_states {hidden_states[-1][0, 0, : 20]}")

