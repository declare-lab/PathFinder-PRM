Qwen_Policy_Prompt = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful math problem solving assistant. Please reason step by step, and put your final answer within \\boxedleft-bracketsright-brackets.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{previous_steps}"""