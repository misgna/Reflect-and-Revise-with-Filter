# Reflect-and-Revise-with-Rubric-Filtering
We augment the Reflect-and-Revise approach with the capability to capture drift and novelty by repurposing conditional entropy to quantify both.

# Reflect-and-Revise
Official implementation of "Automated Refinement of Essay Scoring Rubrics for Language Models via Reflect-and-Revise"

## Prepare dataset
```bash
# download asap https://www.kaggle.com/competitions/asap-aes
./dataset/asap-ets
```
    
```bash
# Refine
python inference.py
```

```bash
# evaluate
python evaluate.py
```
