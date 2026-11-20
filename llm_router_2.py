import torch
import time
from tracker import tracker


async def get_llm_response(messages, model, tokenizer, params, call_type="verifier") -> str:
    start_time = time.perf_counter()
    
    tokenizer.padding_side = "left"
    # Ensure pad token is set for tensor creation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = model.device if hasattr(model, "device") else "cuda"
    # 1. Format prompt into text string or list of text strings
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 2. Tokenize with padding & truncation enabled here
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    # 3. Extract parameter values safely
    max_tokens = params.get("max_tokens", 1024) if isinstance(params, dict) else 1024
    temp = params.get("temperature", 0.7) if isinstance(params, dict) else 0.7

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True if temp > 0 else False,
            temperature=float(temp) if temp > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id
        )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tracker.record_call(call_type=call_type, elapsed_gpu_time=elapsed)
    
    # 4. Decode generated output (skipping prompt input sequence)
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


