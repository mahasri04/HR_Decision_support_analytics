# src/rules_engine.py

"""
A smarter, weighted HR recommendations engine.

Enhancements:
    ✓ Severity-based recommendations (High / Medium / Low)
    ✓ Weighted scoring to prioritize issues
    ✓ Multi-factor HR intelligence (e.g., turnover + compa)
    ✓ Clean, structured output for dashboards
    ✓ Threshold override support
"""

def generate_recommendations(metrics, thresholds=None):
    """
    Generate structured HR recommendations based on KPI metrics.
    
    Args:
        metrics (dict): HR KPI values
        thresholds (dict, optional): custom rule thresholds
    
    Returns:
        list of dict:
            [
              {
                "priority": "High",
                "issue": "High Turnover",
                "recommendation": "Strengthen retention programs..."
              },
              ...
            ]
    """

    # ---------------------------
    # Default Thresholds
    # ---------------------------
    if thresholds is None:
        thresholds = {
            'turnover_high': 12,
            'turnover_moderate': 8,
            'compa_low': 0.95,
            'training_low': 50,
            'performance_low': 3.0,
            'labour_cost_pct_high': 40
        }

    recs = []

    # Helper to add structured rule output
    def add(priority, issue, msg):
        recs.append({
            "priority": priority,
            "issue": issue,
            "recommendation": msg
        })

    t = thresholds  # shorthand
    m = metrics     # shorthand

    # -------------------------------------------------------
    # TURNOVER RULES
    # -------------------------------------------------------
    tr = m.get('turnover_rate', 0)

    if tr > t['turnover_high']:
        add(
            "High",
            "High Turnover",
            "Turnover exceeds safe limits. Implement retention interviews, stay surveys, and targeted compensation fixes."
        )

    elif tr > t['turnover_moderate']:
        add(
            "Medium",
            "Moderate Turnover",
            "Turnover is elevated. Strengthen employee engagement and monitor at-risk departments."
        )

    # -------------------------------------------------------
    # COMPA RATIO RULES
    # -------------------------------------------------------
    compa = m.get("CompaRatio", m.get("compa_ratio", 1.0))

    if compa < t['compa_low']:
        add(
            "High",
            "Below-Market Compensation",
            "Employees are paid below market. Review salary bands and address compression equity issues."
        )

    # -------------------------------------------------------
    # TRAINING COVERAGE RULES
    # -------------------------------------------------------
    train = m.get("training_coverage", 100)

    if train < t['training_low']:
        add(
            "Medium",
            "Low Training Coverage",
            "Training participation is low. Expand learning programs and mandate training for key skill areas."
        )

    # -------------------------------------------------------
    # PERFORMANCE RATING RULES
    # -------------------------------------------------------
    perf = m.get("avg_performance", 5)

    if perf < t["performance_low"]:
        add(
            "Medium",
            "Low Performance Trends",
            "Performance ratings are declining. Introduce coaching and performance improvement plans."
        )

    # -------------------------------------------------------
    # LABOUR COST RULE
    # -------------------------------------------------------
    lc = m.get("labour_cost_pct_of_revenue")

    if lc and lc > t["labour_cost_pct_high"]:
        add(
            "High",
            "Labour Cost Exceeds 40% of Revenue",
            "Optimize staffing, automate repetitive tasks, or rebalance cost centers."
        )

    # -------------------------------------------------------
    # COMBINED MULTI-KPI RULES (ADVANCED)
    # -------------------------------------------------------

    # High turnover + Low compa ratio → compensation-driven attrition
    if tr > t['turnover_high'] and compa < t['compa_low']:
        add(
            "High",
            "Compensation-Driven Attrition",
            "Both turnover and compa ratio are alarming. Employees may be leaving due to below-market pay. Revisit compensation strategy immediately."
        )

    # High turnover + low training → capability-building issue
    if tr > t['turnover_high'] and train < t['training_low']:
        add(
            "Medium",
            "Skill/Training Gaps Causing Attrition",
            "Low training coverage paired with high turnover indicates capability mismatch. Expand upskilling programs to reduce exits."
        )

    # If no issues found
    if not recs:
        add(
            "Low",
            "Stable KPIs",
            "All metrics are within healthy ranges — maintain existing strategy and continue monitoring."
        )

    return recs
