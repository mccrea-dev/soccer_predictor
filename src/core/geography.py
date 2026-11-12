"""Which confederation each national team belongs to, and features built on that.

Elo has a blind spot here. Teams mostly play inside their own confederation, so ratings are well
calibrated within Europe or within Asia but only loosely tied together across them: a European
1800 and an Asian 1800 aren't necessarily the same strength. Knowing the confederations lets the
model measure how far off Elo tends to be for each kind of cross-confederation matchup.

conf_of(name) returns UEFA / CONMEBOL / CONCACAF / CAF / AFC / OFC, or 'OTH' for sides outside
FIFA's six confederations (regional and non-FIFA teams that only show up in friendlies).
"""

CONFS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]

_MEMBERS = {
    "CONMEBOL": {
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru",
        "Uruguay", "Venezuela",
    },
    "UEFA": {
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "England",
        "Estonia", "Faroe Islands", "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
        "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Kazakhstan", "Kosovo", "Latvia",
        "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro", "Netherlands",
        "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal", "Romania", "Russia",
        "San Marino", "Scotland", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Turkey", "Ukraine", "Wales",
    },
    "CONCACAF": {
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda",
        "Bonaire", "British Virgin Islands", "Canada", "Cayman Islands", "Costa Rica", "Cuba",
        "Curaçao", "Dominica", "Dominican Republic", "El Salvador", "French Guiana", "Grenada",
        "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico",
        "Montserrat", "Nicaragua", "Panama", "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Martin", "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
        "Trinidad and Tobago", "Turks and Caicos Islands", "United States",
        "United States Virgin Islands",
    },
    "CAF": {
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde",
        "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo", "Côte d'Ivoire",
        "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia",
        "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia",
        "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
        "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe", "Senegal", "Seychelles",
        "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
        "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    },
    "AFC": {
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia",
        "China PR", "Chinese Taipei", "Taiwan", "Guam", "Hong Kong", "India", "Indonesia", "Iran",
        "Iraq", "Japan", "Jordan", "Korea DPR", "Korea Republic", "Kuwait", "Kyrgyzstan", "Laos",
        "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
        "Northern Mariana Islands", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
        "Saudi Arabia", "Singapore", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Timor-Leste",
        "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
    },
    "OFC": {
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia", "New Zealand",
        "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti", "Tonga", "Vanuatu",
    },
}

_CONF_OF = {team: conf for conf, teams in _MEMBERS.items() for team in teams}


def conf_of(name):
    """Confederation for a team name, 'OTH' if it isn't in any of the six."""
    return _CONF_OF.get(name, "OTH")


def add_conf_features(df, home_adv=100.0):
    """Add confederation features to a match table. Needs home_team, away_team, home_goals,
    away_goals, neutral, elo_h and elo_a; the frame is sorted by date on the way in.

      hc_* / ac_*  one-hot confederation of the home / away team
      cross_conf   1 when the two teams come from different confederations
      conf_bias    for this (home confederation, away confederation) pairing, the average amount
                   by which home teams have beaten or missed their Elo expectation so far

    conf_bias is a running average over earlier matches only: each row reads the average first
    and adds its own result afterwards, so a match never sees its own outcome.
    """
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    df = df.sort_values("date").reset_index(drop=True)
    hc = df["home_team"].map(conf_of).to_numpy()
    ac = df["away_team"].map(conf_of).to_numpy()
    eh = df["elo_h"].to_numpy(float); ea = df["elo_a"].to_numpy(float)
    neu = df["neutral"].to_numpy(int)
    hg = pd.to_numeric(df["home_goals"], errors="coerce").to_numpy(float)
    ag = pd.to_numeric(df["away_goals"], errors="coerce").to_numpy(float)
    for c in CONFS:
        df[f"hc_{c}"] = (hc == c).astype(float)
        df[f"ac_{c}"] = (ac == c).astype(float)
    df["cross_conf"] = (hc != ac).astype(float)
    # running sum and count of (actual - expected) per confederation pairing
    s = defaultdict(float); cnt = defaultdict(int)
    bias = np.full(len(df), np.nan)
    for i in range(len(df)):
        # Elo's expected score for the home side (same formula as compute_elo)
        we = 1.0 / (10 ** (-(((eh[i] + (0.0 if neu[i] else home_adv)) - ea[i])) / 400.0) + 1.0)
        key = (hc[i], ac[i])
        if cnt[key] > 0:                      # read the average before this match is added
            bias[i] = s[key] / cnt[key]
        if not (np.isnan(hg[i]) or np.isnan(ag[i])):    # unplayed fixtures don't update anything
            w = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
            s[key] += (w - we); cnt[key] += 1
    df["conf_bias"] = bias
    return df


CONF_FEATURES = ([f"hc_{c}" for c in CONFS] + [f"ac_{c}" for c in CONFS]
                 + ["cross_conf", "conf_bias"])
