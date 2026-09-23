# OpenRouter direct-VLM benchmark audit

## Verdict

The image/data/evaluation pipeline is internally correct. The reported GPT-5.4 result—737/1,899 Top-1 (38.81%) and 1,203/1,899 Top-5 (63.35%)—recomputes exactly from the saved rows.

It should be named **GPT-5.4 direct closed-set, single-pass, minimal-reasoning, fixed-candidate-order baseline**. It is not evidence of GPT-5.4's maximum possible botanical accuracy. The fixed candidate order is a real order-sensitivity limitation, and the attempted high-reasoning audit was inconclusive because the completion cap truncated 51 of 63 responses.

## Evidence chain

| Link | Independent check | Result |
|---|---|---:|
| Dataset | Unique image IDs / filenames | 1,899 / 1,899 |
| Decode | Images decoded and re-encoded in Modal | 1,899 / 1,899 |
| Payload | JPEG RGB, metadata removed, exact base64 round-trip | 1,899 / 1,899 |
| Dimensions | Rebuilt payload vs saved benchmark dimensions | 0 mismatches |
| Candidates | Unique scientific / vernacular labels | 85 / 85 |
| Requests | Successful model-image rows | 3,798 / 3,798 |
| Model routing | Saved served model | openai/gpt-5.4 on 1,899 rows |
| Responses | Five distinct, in-list classes and consistent first rank | 3,798 / 3,798 |
| Traceability | Unique OpenRouter generation IDs | 3,798 / 3,798 |
| Ground truth | GPT file/truth rows vs the earlier BioCLIP export | 0 / 0 mismatches |
| Scoring | Recomputed GPT correctness vs saved flag | 0 mismatches |

The outbound API message contains only the fixed candidate prompt and the base64 image. `image_id`, file name, local path, organ label and ground truth remain local and are not inserted into the message. Ground truth is loaded after all requested model calls finish.

## Why a large BioCLIP gap is plausible

- BioCLIP: 1472/1,899 Top-1 (77.51%).
- GPT-5.4: 737/1,899 Top-1 (38.81%).
- Paired outcomes: both correct 672; GPT only 65; BioCLIP only 800; both wrong 362.
- GPT is far above random guessing (1/85 = 1.18%) and reaches 63.35% Top-5. It often identifies a close candidate without resolving the exact species.
- The performance gradient follows visual difficulty: fruit 53.1%, inflorescence 47.4%, leaf 41.9%, twig 28.3%, bark 17.3%.

## Protocol limitations found by the audit

### Candidate-order sensitivity

The full run used one deterministic randomized candidate order to enable prompt caching. GPT selected positions 1–20 for 44.9% of images, while only 23.5% of ground-truth records belonged to those positions.

An earlier pilot used a different deterministic order for every image. Among 56 images with valid outputs in both pilots, predictions agreed only 60.7%. Top-1 was 28.6% with per-image shuffle and 32.1% with fixed order. This confirms order sensitivity, but the observed pilot shift is much smaller than the 38.7 percentage-point BioCLIP gap.

### High-reasoning audit

High reasoning with original image detail completed only 12/63 outputs before the 2,000-token completion cap. Successful requests averaged 88.0s. On those same 12 images, high reasoning got 7 Top-1 versus 5 for minimal reasoning. This is directional evidence only; the completed subset is small and non-random, so no accuracy claim should be made.

## Example-level evidence

| image_id | Truth | GPT-5.4 | Rank of truth | BioCLIP | Interpretation |
|---|---|---|---:|---|---|
| baskauf/38299 | Pinus rigida | Pinus strobus | 3 | correct | Clear pine image; GPT reaches the right genus but misses the species. |
| baskauf/66166 | Gleditsia triacanthos | Robinia pseudoacacia | 2 | correct | Compound-leaf/inflorescence lookalikes; truth is GPT's second choice. |
| baskauf/50998 | Ginkgo biloba | correct | 1 | wrong | GPT contributes genuine wins that BioCLIP misses. |
| baskauf/89954 | Platanus occidentalis | correct | 1 | correct | Both systems succeed on a distinctive whole-tree view. |

## Recommended reporting language

Report the 38.81% result as a valid **cost-controlled direct-VLM baseline** under the exact protocol above. Do not describe it as GPT-5.4's botanical ceiling. A stronger follow-up should balance candidate order across several fixed permutations and use a completion budget sized for the selected reasoning effort.
