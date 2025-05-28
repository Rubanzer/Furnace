import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Furnace Mix Optimiser", layout="wide")
st.title("Furnace Charge Mix Optimiser")

# ──────────────────────────────────────────────────────────────────────────────
# 1. MATERIAL SETUP & EDITABLE PRICE TABLE
# ──────────────────────────────────────────────────────────────────────────────
materials = ["Scrap", "Pellet DRI", "DR-cello DRI", "Lump DRI"]
default_rates   = [30000, 25000, 24000, 23500]          # ₹/t
default_yields  = [0.95  , 0.84  , 0.89  , 0.87  ]      # t_liquid / t_charge
default_p       = [0.030 , 0.080 , 0.035 , 0.080 ]      # %

if "rate_df" not in st.session_state:
    st.session_state["rate_df"] = pd.DataFrame(
        {"Material": materials, "Rate (₹/t)": default_rates}
    )

price_editor = st.data_editor(
    st.session_state["rate_df"],
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    key="rate_editor",
)
st.session_state["rate_df"] = price_editor
rates = dict(zip(price_editor["Material"], price_editor["Rate (₹/t)"].astype(float)))

# ──────────────────────────────────────────────────────────────────────────────
# 2. SIDEBAR INPUTS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Material Yields & Phosphorus (%)")
yields, ps = {}, {}
for i, m in enumerate(materials):
    yields[m] = st.sidebar.number_input(f"{m} yield", value=default_yields[i], step=0.01, format="%.4f")
    ps[m]     = st.sidebar.number_input(f"{m} phosphorus %", value=default_p[i],   step=0.001, format="%.3f")

st.sidebar.header("Fixed Plant Numbers")
tap_t  = st.sidebar.number_input("Tonnes per heat (t/heat)",              value=40)
bmin   = st.sidebar.number_input("Heat Time (min @ 100 % scrap)",    value=80)
kmin   = st.sidebar.number_input("Additional time per percent of Sponge (min / +1 % sponge)",    value=1.0)
q80    = st.sidebar.number_input("Tonnes per month at 100% Scrap (t/month @ 80-min heats)",value=23000)
cu     = st.sidebar.number_input("Electricity cost ₹/kWh",      value=7.5, step=0.1)
F      = st.sidebar.number_input("Fixed overhead / month (₹)",  value=75_000_000, step=1_000_000)
p_max  = st.sidebar.number_input("Phosphorus limit (%)",        value=0.050, format="%.3f")
s_min  = st.sidebar.number_input("Minimum scrap fraction",      value=0.60,  format="%.2f")

st.sidebar.header("Power Model")
bkwh  = st.sidebar.number_input("bkwh (kWh/heat @ 100 % scrap)", value=22800.0, step=100.0)
kkwh  = st.sidebar.number_input("kkwh (extra kWh per +1 % sponge)", value=45.0, step=1.0)

# ──────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def kwh_per_heat(sponge: float) -> float:
    """kWh consumed per heat given sponge fraction (1 – scrap)."""
    return (bkwh + (kkwh * 100.0 * sponge))

def power_r_t(sponge: float) -> float:
    """₹/t electricity cost at this sponge level."""
    return ((kwh_per_heat(sponge) / tap_t) * cu)

def material_r_t(xs, xp, xc, xl) -> float:
    """₹/t material cost with proper yield adjustment."""
    ys, yp, yc, yl = (yields["Scrap"], yields["Pellet DRI"], yields["DR-cello DRI"], yields["Lump DRI"])
    rs, rp, rc, rl = (rates["Scrap"],  rates["Pellet DRI"],  rates["DR-cello DRI"],  rates["Lump DRI"])
    return (
        ((xs * rs) + (xp * rp) + (xc * rc) + (xl * rl))
        /
        ((xs * ys) + (xp * yp) + (xc * yc) + (xl * yl))
    )

def cost(vars):
    """Objective = ₹/t (material + power + fixed)."""
    xs, xp, xc, xl = vars
    sponge = (1.0 - xs)

    # Heat length & monthly output
    heat_minutes   = (bmin + (kmin * 100.0 * sponge))
    tons_per_month = ((q80 * bmin) / heat_minutes)

    fixed_r_t  = (F / tons_per_month) if tons_per_month else np.inf
    return (
        material_r_t(xs, xp, xc, xl)
        + power_r_t(sponge)
        + fixed_r_t
    )

def avg_p(vars):
    xs, xp, xc, xl = vars
    return (
        (xs * ps["Scrap"])
        + (xp * ps["Pellet DRI"])
        + (xc * ps["DR-cello DRI"])
        + (xl * ps["Lump DRI"])
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4. OPTIMISATION SET-UP
# ──────────────────────────────────────────────────────────────────────────────
cons = [
    {"type": "eq",   "fun": lambda v: (np.sum(v) - 1.0)},  # fractions sum to 1
    {"type": "ineq", "fun": lambda v: (v[0] - s_min)},     # ≥ minimum scrap
    {"type": "ineq", "fun": lambda v: (p_max - avg_p(v))}, # ≤ P-limit
]
bounds = [(0.0, 1.0)] * 4
rng = np.random.default_rng()

def random_seed():
    """Start every attempt at exactly s_min scrap, random rest."""
    r   = rng.random(3)
    r  /= r.sum()
    xs  = s_min
    xp, xc, xl = (r * (1.0 - s_min))
    return [xs, xp, xc, xl]

# ──────────────────────────────────────────────────────────────────────────────
# 5. RUN OPTIMISER WITH AUTO-RETRY
# ──────────────────────────────────────────────────────────────────────────────
max_attempts = 30
status       = st.empty()
res          = None

for attempt in range(1, max_attempts + 1):
    status.info(f"Optimising… attempt {attempt}/{max_attempts} ⏳")
    x0  = random_seed()
    res = minimize(cost, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    if res.success:
        status.success(f"Optimisation converged after {attempt} attempt(s) ✔️")
        break
else:
    status.error("Failed to converge after maximum attempts. Showing best-found mix.")

# ──────────────────────────────────────────────────────────────────────────────
# 6. RESULTS – MIX TABLE & COST BREAKDOWN
# ──────────────────────────────────────────────────────────────────────────────
xs, xp, xc, xl = res.x
mix_df = pd.DataFrame({
    "Material": materials,
    "Fraction":                [xs, xp, xc, xl],
    "Tonnage (t/heat)": (np.array([xs, xp, xc, xl]) * tap_t),
})
mix_df["Fraction"]          = mix_df["Fraction"].round(4)
mix_df["Tonnage (t/heat)"]  = mix_df["Tonnage (t/heat)"].round(2)

sponge          = (1.0 - xs)
heat_minutes    = (bmin + (kmin * 100.0 * sponge))
tons_per_month  = (q80 * (bmin / heat_minutes))
kwh_heat        = kwh_per_heat(sponge)
fixed_r_t_val   = (F / tons_per_month) if tons_per_month else np.inf
power_r_t_val   = power_r_t(sponge)
mat_r_t_val     = material_r_t(xs, xp, xc, xl)
final_cost      = (mat_r_t_val + power_r_t_val + fixed_r_t_val)

breakdown_df = pd.DataFrame({
    "Component": ["Material", "Power", "Fixed"],
    "₹/t":       [mat_r_t_val, power_r_t_val, fixed_r_t_val],
})

# ──────────────────────────────────────────────────────────────────────────────
# 7. DISPLAY
# ──────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Optimal Mix")
    st.dataframe(mix_df, use_container_width=True)

with col2:
    st.subheader("Cost Breakdown (₹/t)")
    st.dataframe(breakdown_df, use_container_width=True)
    st.metric("Total Cost (₹/t)",      f"{final_cost:,.0f}")
    st.metric("Heat time (min)",       f"{heat_minutes:.1f}")
    st.metric("kWh per heat",          f"{kwh_heat:.0f}")
    st.metric("Monthly output (t)",    f"{tons_per_month:,.0f}")

st.caption(
    "Enter electricity parameters in the sidebar. "
    "Solver auto-retries until a valid solution meets ≥ {:.0f}% scrap & ≤ {:.3f}% P."
    .format(s_min * 100, p_max)
)