# Should We Stop Giving Discounts?

## The Real Story

During my US Cellular internship, I kept seeing us throw discounts at customers threatening to cancel. It felt like we were training people to complain for freebies. So I wanted to know: **does this actually work, or are we just burning money?**

I can't share real customer data (obviously), but this analysis recreates the patterns I saw in 2M+ customer records. The code simulates what would happen if we ran a targeted 15% discount campaign for high-risk customers.

**The punchline:** Depends on your retention uplift. If 40% of at-risk customers actually stay because of the discount, you come out ahead. If it's lower, you're better off letting them walk.

---

## What This Code Does

1. **Generates realistic telecom customers** based on actual churn drivers (contract type, support calls, payment history, tenure)
2. **Trains 3 ML models** to predict who's about to leave (Gradient Boosting wins at ~0.85 AUC)
3. **Simulates a discount campaign** targeting high-risk customers
4. **Calculates the ROI** — revenue saved vs discount cost

You get two charts + a CSV with the full breakdown.

---

## Files

| File | What |
|---|---|
| `churn_discount_analysis.py` | The full analysis |
| `churn_drivers.png` | 6-panel EDA showing what makes people leave |
| `model_comparison.png` | ROC curves for all 3 models |
| `discount_analysis.csv` | Baseline vs discount campaign numbers |
| `README.md` | This file |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python churn_discount_analysis.py
```

Takes about 10 seconds. You'll see:
- Risk distribution across your customer base
- Model performance comparison
- The ROI calculation with exact dollar amounts

---

## The Key Insight

**Month-to-month contracts** are where you hemorrhage customers. Churn rate is 3x higher than annual contracts. If I were still there, I'd push harder to convert people to longer commitments — way more effective than discount band-aids.

Also: **support call volume spikes right before someone churns**. We should be flagging customers who suddenly start calling as high-risk, not waiting for them to threaten cancellation.

---

## What I'd Change for Production

If this were going into a real retention system:

1. Use actual customer data with real discount acceptance rates
2. Add uplift modeling to predict *who responds to discounts* (not everyone does)
3. Test different discount levels (10%, 15%, 20%) — there's probably a sweet spot
4. Track how long saved customers actually stick around after the discount ends
5. Build this into a monthly automated pipeline that flags accounts for the retention team

---

## Why This Matters for My Portfolio

Most data science projects are just "look I can train a model with 90% accuracy!" This one answers a real business question with a dollar amount attached. That's what analytics is supposed to do.
