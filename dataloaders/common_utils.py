def alignment_dpo_format(tokenizer, samples):
    # Format system
    if len(samples['system']) > 0:
        message = {"role": "system", "content": samples['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": samples['text']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = samples['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = samples['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def alignment_kto_format(tokenizer, samples):
     # Format system
    if len(samples['system']) > 0:
        message = {"role": "system", "content": samples['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": samples['text']}
    prompt = tokenizer.apply_chat_template([message], 
                                           tokenize=False, 
                                           add_generation_prompt=True)

    completion = samples['completion'] + "<|im_end|>\n"
    return {
        "prompt": system + prompt,
        "completion": completion,
        "label": samples["label"],
    }
