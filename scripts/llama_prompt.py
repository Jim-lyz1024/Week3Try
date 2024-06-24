from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

def generate_text(prompt, max_length=1000, num_return_sequences=1):
    # 对输入的文本进行编码
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成文本
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences,
        do_sample=True,  
        top_k=50,        
        top_p=0.95       
    )
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

prompt = "What is a Queen?"

generated_texts = generate_text(prompt, max_length=100, num_return_sequences=3)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}:\n{text}\n")
