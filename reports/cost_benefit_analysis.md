# Cost-Benefit Analysis (BR-04)

Simulated on the untouched hold-out set (1,409 customers). All business numbers are **assumptions from BRD section 8**, not observed values.

**Net benefit** = (offer success rate x 12 x MonthlyCharges of each churner we reached) - (offer cost x customers contacted).

## 1. Assumptions
Offer cost **$20** per contact; **30%** of contacted churners accept; customer value = **12 x MonthlyCharges** (average $769 in the hold-out).

## 2. Results at the BRD assumptions
| Strategy | Customers contacted | Campaign cost (USD) | Net benefit (USD) | Gain vs random (same count) |
|---|---|---|---|---|
| Contact everyone | 1409 | 28,180 | 69,794 | - |
| Random, same count as High tier | 584 | 11,680 | 28,928 | - |
| Model - High tier (t=0.26, operating) | 584 | 11,680 | 69,552 | +40,624 |
| Model - cost-optimal (t=0.08) | 942 | 18,840 | 75,812 | +29,151 |

## 3. How the threshold was chosen
* **Cost-optimal threshold (0.08)** maximises net benefit on training out-of-fold predictions. It is low because one saved customer (about $231 expected) is worth far more than one offer ($20); the break-even churn probability for an average customer is only about **8.7%**.
* **Operating threshold (0.26)** maximises net benefit **among thresholds that also meet the BRD recall (>= 0.7) and precision (>= 0.5) targets**. It defines the High tier.
* The BRD asks for both a cost-based threshold and recall/precision targets. Under these assumptions the two disagree, so both are reported: the operating threshold satisfies the acceptance criteria, the cost-optimal one maximises money.

## 4. What this means for the sponsor (important)
* Targeting beats **random targeting of the same size** by **$40,624** (BRD criterion: positive and quantified - met).
* But at these assumptions **contacting every customer is almost as profitable** as contacting only the High tier ($69,794 vs $69,552), because offers are cheap relative to customer value. The cost-optimal threshold beats contacting everyone by $6,018.
* The model's value therefore comes from **(a) spending $11,680 instead of $28,180** for nearly the same return, **(b) limited call-centre capacity or budget**, and **(c) higher offer costs** - see the sensitivity table. Finance should confirm the real offer cost and success rate before the campaign is sized.

## 5. Sensitivity (hold-out net benefit, USD)
| offer_cost | success_rate | contact_everyone | random_same_count_as_high_tier | high_tier_only | model_expected_value_rule |
|---|---|---|---|---|---|
| 20 | 0.100 | 4,478 | 1,856 | 15,397 | 15,778 |
| 20 | 0.200 | 37,136 | 15,392 | 42,475 | 45,713 |
| 20 | 0.300 | 69,794 | 28,928 | 69,552 | 77,054 |
| 50 | 0.100 | -37,792 | -15,664 | -2,123 | 4,072 |
| 50 | 0.200 | -5,134 | -2,128 | 24,955 | 25,899 |
| 50 | 0.300 | 27,524 | 11,408 | 52,032 | 53,691 |
| 100 | 0.100 | -108,242 | -44,864 | -31,323 | 146 |
| 100 | 0.200 | -75,584 | -31,328 | -4,245 | 8,144 |
| 100 | 0.300 | -42,926 | -17,792 | 22,832 | 27,735 |
| 150 | 0.100 | -178,692 | -74,064 | -60,523 | 0 |
| 150 | 0.200 | -146,034 | -60,528 | -33,445 | 916 |
| 150 | 0.300 | -113,376 | -46,992 | -6,368 | 12,216 |

`model_expected_value_rule` contacts a customer only if churn probability x success rate x customer value > offer cost (a per-customer cost-benefit rule). Full table: `cost_benefit_sensitivity.csv`.
