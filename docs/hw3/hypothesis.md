# HW3 hypothesis and predictions (written before any HW3 result)

## Limitation in the HW2 model

The HW2 probe sees one vector per region: the **mean** of Qwen2.5-VL's layer-21 tokens under the box. Every token gets the same weight 1/L, whether it covers the object or the background around it. HW2's weakest category is **person (50% test recall)**. Person boxes often consist mostly of a shirt, a face or an arm, and their mean vector sits between person, clothing, head and face, and limb.

## Design

1. **Attention pooling replaces the mean.** A learned query scores each token under the box, and a softmax over the box turns the scores into weights. The pooled vector is the weighted average of the tokens. The 8-head variant gives each head its own query and its own 448-number slice of the vector.
2. **The MLP gets a second hidden layer:** 3584 → 512 → ReLU → 512 → ReLU → 20.
3. **The queries start at zero.** Every token then starts with the same weight 1/L, so the model starts as exactly the HW2 mean pool, and any difference has to be learned.

## Hypothesis

Learned pooling lets the probe up-weight the tokens that show the object and down-weight background tokens. This should raise accuracy most where the mean mixes the most background into the vector. The second hidden layer adds non-linearity on top of the pooled vector.

## Predictions, to be checked against the results

1. **Attention cannot change anything on the 18.5% of regions with one token.** There a softmax over one token always gives weight 1. Any accuracy change there would come from the MLP depth or seed noise, not from attention.
2. **Any gain from attention should appear for boxes with 10 or more tokens,** and grow with the token count.
3. **Person, limb and clothing should gain the most recall.** These are the categories whose boxes mix several kinds of content.
4. **A second hidden layer on the mean pool should add overfitting, not accuracy.** HW2 already overfits by about epoch 8, with 87% train accuracy against 73% on validation. This prediction tests the "more non-linearity" motivation on its own.
5. **One head against eight:** the research repo's `planning.md` (item 5) records that, on layer-21 probes, a one-head attention pool was not reliably better than the mean while eight heads were. We therefore expect 8 heads to be at least as good as 1 head.

## How it could fail

- **Attention may never leave the mean pool.** With zero queries the keys get no gradient on the first step, and early stopping ends training around epoch 8. Normalized attention entropy is logged every epoch, where 1.0 means uniform weights, to detect this.
- **A single query may lock onto a token position instead of the content.** The research `CHANGELOG.md` (2026-06-19) records a single-query attention pool that locked onto position-consistent tokens and called a horse "a dog". We measure how often the top-weighted token is the box's first token, against the chance rate 1/L.
- **Weight decay pulls the queries back toward zero,** which keeps attention near uniform.
- **A few tokens with very large values may dominate the scores.** Layer 21 has outlier dimensions near −80.
- **The MLP's input statistics are fixed from the mean pool.** If attention sharpens, the pooled vectors change scale and the standardization no longer fits.
- **More parameters on 40,000 regions may simply overfit more.**

## Controls

Six arms train with the same split, data and budget: pooling {mean, attention with 1 head, attention with 8 heads} crossed with depth {1 hidden layer, 2 hidden layers}.

- **The budget:** Adam with learning rate {3e-4, 1e-3} × weight decay {1e-4, 1e-3}, dropout 0.5, batch 256, and early stopping on validation loss.
- **Selection:** each arm keeps its best setting on validation accuracy, then reruns that setting with 3 seeds. The test set is scored only for the selected setting.
- **Effects:** the attention effect and the depth effect are read as paired-bootstrap contrasts on seed-averaged test predictions.

## Originality, stated plainly

Attention pooling over a set of vectors is an existing component. It appears in attention-based multiple-instance learning, and the research project used an attention-pool connector. The design choices here are what is specific to this data: tying the hypothesis to box contents, starting from the exact mean pool, and testing the predictions above by box size, by category and by attention statistics.
