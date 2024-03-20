def generate_gemma_prompt(original_text, rewritten_text, rewrite_prompt=None):
    instruction_text = 'Generate a rewrite_prompt that effectively transforms the provided original_text into the provided rewritten_text. The rewrite_prompt must be clearly explain how to the original_text is transformed to the rewritten_text, focus on explaining the changes of tone, writting style, publishing, etc. Keep the rewrite_prompt concise, less than 100 words.'
    
    text = f"""<start_of_turn>user {instruction_text}
Here is the given texts:
# original_text:
{original_text}

# rewritten_text:
{rewritten_text}
<end_of_turn>
<start_of_turn>model""" + \
    ("""\n{rewrite_prompt} <end_of_turn>""" if rewrite_prompt else '')
    
    return text