# Reflect-and-Revise-with-Rubric-Filtering
We have extended the Reflect-and-Revise in order to filter rubric for drift and novelty. We re-purpose conditional entropy for quantifying structural drift and novelty.

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
