
    max_value = max (val.data for val in logits)
    exps = [(val -max).exp() for val in logits]
    total = sum(exps)
    return [ e/ total for e in exps]