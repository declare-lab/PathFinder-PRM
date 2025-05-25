CLS_PROMPT = """
You are an analytical math instructor grading a student's work. Think step-by-step through your analysis. Below is the math question, previous steps by the student, and current step to evaluate.

{context}

Your task is to rigorously examine the current step and determine if it contains ANY mathematical errors. Assign binary scores (0 = wrong, 1 = correct) based on three criteria:

A) Mathematical logic — Is the current step, **on its own**, mathematically valid? Check for:
   • Calculation errors
   • Incorrect formula application 
   • Invalid operations or simplifications
   • Algebra mistakes or sign errors
   • Incorrect assertions

B) Consistency — Is the current step logically consistent with:
   • Established ground truth
   • Previous steps
   • Any constraints or conditions established earlier
   • The mathematical domain applicable to this problem

C) Simplicity and optimality — is this step an efficient next step toward the solution? Check for:
   • Redundant statements: factually correct statements that do not help progress toward the solution.
   • Circular logic: does this step come to a conclusion already previously established?
   • Non-clarity: Are the assertions made in this step ambiguous in a way that obsfucates their purpose?
   • Optimality: is the **idea** of this step the near optimal approach one would take to solve the problem?


Double check all listed criterion here explicitly in your reasoning. In your analysis, be sensitive to subtle issues like missing pre-requisites/assumptions, correct-looking statements with slight errors and high confidence statements containing errors.

IMPORTANT POINTS: 

- If you find ANY error, even a minor one, you MUST assign a score of 0 to the appropriate criteria. Be skeptical and verify all claims thoroughly.

- For incorrect steps, wherever possible, attempt to categorize the issue as violating **one of the three criterion** (i.e., assign score 0 to **only one category**). Assign multiple 0 scores only for serious errors.

You must format your answer as below:

Reasoning:
{{Provide detailed analysis, showing all verification steps and explicitly identifying any errors found}}

Final answers:
Score A: 
{{0 or 1 only}}
Score B: 
{{0 or 1 only}}
Score C:
{{0 or 1 only}}


"""