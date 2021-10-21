import numpy as np
import tensorflow as tf
import copy
import pandas as pd


def isnip(model, dataset, num_batch=1, pruning_ratio=0.01, num_step=10,  mode='exp'):
    if mode == 'linear':
        pruning_steps = [
            1 - ((x + 1) * (1 - pruning_ratio) / num_step) for x in range(num_step)]

    elif mode == 'exp':
        pruning_steps = [np.exp(
            0 - ((x + 1) * (0 - np.log(pruning_ratio)) / num_step)) for x in range(num_step)]
    grads = []
    weights, masks = model.get_pruneable_variables_with_masks()
    weights_values = [w.numpy() for w in weights]
    pre_mask = None
    for perc in pruning_steps:
        gradients = _evalute_loss_gradients(model, dataset, weights, num_batch)
        saliencies = [np.abs(g * w) for g, w in zip(gradients, weights_values)]
        if pre_mask is not None:
            min_saliency = _get_minimum_salicency(saliencies)
            for i in range(len(saliencies)):
                saliencies[i][~pre_mask[i]] = min_saliency
        mask_flag = _get_mask(saliencies, perc)
        _assign_value(masks, mask_flag)
        pre_mask = copy.deepcopy(mask_flag)
    _print_mask_info(pre_mask)

def igrasp(model, dataset, num_batch=1, pruning_ratio=0.01, num_step=10,  mode='exp'):
    """
        the implemention follows the GraSP approximation in Force,
        see at https://github.com/naver/force/blob/master/pruning/pruning_algos.py
    """
    
    if mode == 'linear':
        pruning_steps = [
            1 - ((x + 1) * (1 - pruning_ratio) / num_step) for x in range(num_step)]

    elif mode == 'exp':
        pruning_steps = [np.exp(
            0 - ((x + 1) * (0 - np.log(pruning_ratio)) / num_step)) for x in range(num_step)]
    grads = []
    weights, masks = model.get_pruneable_variables_with_masks()
    pre_mask = None
    for perc in pruning_steps:
        gradients = _evalute_loss_gradients(model, dataset, weights, num_batch)
        saliencies = [np.abs(g * g) for g in gradients]
        if pre_mask is not None:
            min_saliency = _get_minimum_salicency(saliencies)
            for i in range(len(saliencies)):
                saliencies[i][~pre_mask[i]] = min_saliency
        mask_flag = _get_mask(saliencies, perc)
        _assign_value(masks, mask_flag)
        pre_mask = copy.deepcopy(mask_flag)
    _print_mask_info(pre_mask)



def synflow(model, dataset, num_batch=1, pruning_ratio=0.01, num_step=10,  mode='exp'):
    if mode == 'linear':
        pruning_steps = [
            1 - ((x + 1) * (1 - pruning_ratio) / num_step) for x in range(num_step)]

    elif mode == 'exp':
        pruning_steps = [np.exp(
            0 - ((x + 1) * (0 - np.log(pruning_ratio)) / num_step)) for x in range(num_step)]
    grads = []
    weights, masks = model.get_pruneable_variables_with_masks()
    weights_values = [w.numpy() for w in weights]
    embeddings = model.get_embeddings()
    embeddings_values = [w.numpy() for w in embeddings]

    # abs all weights
    for w in weights:
        w.assign(np.abs(w.numpy()))
    # set embeddings to all ones
    for w in embeddings:
        w.assign(np.ones_like(w.numpy()))
    fake_dataset = _fake_all_ones_dataset(dataset, num_batch)

    pre_mask = None
    for perc in pruning_steps:
        gradients = _evaluate_logit_gradients(model, fake_dataset, weights, num_batch)
        saliencies = [np.abs(g * w) for g, w in zip(gradients, weights_values)]
        mask_flag = _get_mask(saliencies, perc)
        _assign_value(masks, mask_flag)
        pre_mask = copy.deepcopy(mask_flag)
    _assign_value(masks, pre_mask)
    _print_mask_info(pre_mask)

    # recover variables
    _assign_value(weights, weights_values)
    _assign_value(embeddings, embeddings_values)


def proposed_method(basemethod, feature_maps,
        model, dataset, num_batch=1, pruning_ratio=0.01, num_step=10,  mode='exp'
    ):
    weights, masks = model.get_pruneable_variables_with_masks()
    input_weights, input_mask = weights[0], masks[0]
    input_weights_values = input_weights.numpy()

    # apply base pruning method
    if basemethod == 'isynflow':
        synflow(model, dataset, num_batch, pruning_ratio, num_step,  mode)
    elif basemethod == 'isnip':
        isnip(model, dataset, num_batch, pruning_ratio, num_step,  mode)
    elif basemethod == 'igrasp':
        igrasp(model, dataset, num_batch, pruning_ratio, num_step,  mode)

    mask_val = input_mask.numpy()
    total = mask_val.flatten().sum()
    output_units = mask_val.sum(axis=0)
    out_probes = output_units / output_units.sum()
    input_probes = sum([[feature_maps[col]] * model.embedding_size for col in model.cat_nuniq] + 
        [[feature_maps[col]] * model.embedding_size for col in model.num_cols], [])
    input_probes = _normalize_probs(input_probes)

    selected = 0
    parallel = 10
    curr_mask = np.zeros_like(mask_val)
    while selected < total:
        cur_size = parallel if total - parallel >= selected else (total - selected)
        selected_index = np.random.choice(len(input_probes), int(cur_size), p=input_probes)
        out_selected_index = np.random.choice(len(out_probes), int(cur_size),p=out_probes)
        curr_mask[selected_index, out_selected_index] = 1
        selected = curr_mask.flatten().sum()
    _assign_value(input_mask, curr_mask)


def _evaluate_logit_gradients(model, dataset, variables, num_batch=1):
    grads = []
    for (i, batch) in enumerate(dataset):
        if i >= num_batch:
            break
        with tf.GradientTape() as tape:
            predict = model.evaluate_logit(batch)
        grads.append([v.numpy() for v in tape.gradient(predict, variables)])
    if len(grads) == 1:
        return grads[0]
    else:
        return [np.mean(np.stack([g[i] for g in grads], axis=0), axis=0)
                           for i in range(len(variables))]


def _normalize_probs(probes):
    sum_val = sum(probes)
    return [p / sum_val for p in probes]

def _evalute_loss_gradients(model, dataset, variables, num_batch=1):
    grads = []
    for (i, batch) in enumerate(dataset):
        if i >= num_batch:
            break
        with tf.GradientTape() as tape:
            loss = model.do_train(batch)
        grads.append([v.numpy() for v in tape.gradient(loss, variables)])
    if len(grads) == 1:
        return grads[0]
    else:
        return [np.mean(np.stack([g[i] for g in grads], axis=0), axis=0)
                           for i in range(len(variables))]

def _get_minimum_salicency(saliency):
    "Compute minimum value of saliency globally"
    flattened_saliency = np.concatenate([x.flatten() for x in saliency])
    return flattened_saliency.min()

def _assign_value(variables, values):
    if isinstance(variables, (list, tuple)):
        assert len(variables) == len(values)
        for m,v in zip(variables, values):
            m.assign(v)
    else:
        variables.assign(values)

def _get_mask(saliency, pruning_factor):
    all_scores = np.concatenate([x.flatten() for x in saliency])
    num_params_to_keep = int(len(all_scores) * pruning_factor)
    threshold = all_scores[all_scores.argsort()[-num_params_to_keep]]
    acceptable_score = threshold
    keep_masks = []
    for m in saliency:
        keep_masks.append((m >= acceptable_score))
    return keep_masks


def _print_mask_info(mask_vals):
    activate_weights = []
    if isinstance(mask_vals, dict):
        for key in sorted(mask_vals.keys()):
            total_size = mask_vals[key].size
            non_zero = mask_vals[key].sum()
            activate_weights.append(non_zero)
            print("[WEIGHT] layer:{} total:{} non_zero:{:.0f} nz_ratio:{:.2f}%".format(key, total_size, non_zero, 100 * float(non_zero)/total_size))
    else:
        for key in range(len(mask_vals)):
            total_size = mask_vals[key].size
            non_zero = mask_vals[key].sum()
            activate_weights.append(non_zero)
            print("[WEIGHT] layer:{} total:{} non_zero:{:.0f} nz_ratio:{:.2f}%".format(key, total_size, non_zero, 100 * float(non_zero)/total_size))
    return activate_weights

def _fake_all_ones_dataset(dataset, num_batch=1):
    batch_size = dataset._batch_size.numpy()
    num_cols = len(dataset.element_spec)
    columns = list(dataset.element_spec.keys())
    fake_df = pd.DataFrame(np.ones((batch_size * num_batch, num_cols), dtype=int), columns=columns)
    ds_train = tf.data.Dataset.from_tensor_slices(dict(fake_df)).batch(batch_size)
    return ds_train