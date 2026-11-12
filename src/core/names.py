"""Matching team names across data sources.

Every source spells countries its own way: "Korea Republic" vs "South Korea", "IR Iran" vs
"Iran", "Cabo Verde" vs "Cape Verde". If a name doesn't match, the join doesn't fail loudly, it
just quietly attaches nothing (or worse, the wrong team), and the model trains on garbage.

So the lookup goes in order of how much I trust it: a hand-written alias table first, then an
exact match, and only then a strict fuzzy match. When nothing matches confidently the name is
returned unchanged, because treating a team as unknown is much cheaper than merging two teams.
"""
import re
import difflib
import unicodedata

# Variant spelling (lowercase, no accents) -> the name used everywhere in this project.
# When a new spelling shows up, add it here rather than loosening the fuzzy match below.
ALIASES = {
    "south korea": "Korea Republic", "korea republic": "Korea Republic", "korea south": "Korea Republic",
    "north korea": "Korea DPR", "korea dpr": "Korea DPR",
    "iran": "Iran", "ir iran": "Iran",
    "china": "China PR", "china pr": "China PR",
    "usa": "United States", "united states of america": "United States", "united states": "United States",
    "ivory coast": "Côte d'Ivoire", "cote d'ivoire": "Côte d'Ivoire", "côte d'ivoire": "Côte d'Ivoire",
    "cape verde": "Cabo Verde", "cabo verde": "Cabo Verde",
    "czech republic": "Czechia", "czechia": "Czechia",
    "bosnia": "Bosnia and Herzegovina", "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "drc": "DR Congo", "congo dr": "DR Congo", "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo", "congo-kinshasa": "DR Congo",
    "republic of ireland": "Ireland", "ireland": "Ireland",
    "türkiye": "Turkey", "turkiye": "Turkey", "turkey": "Turkey",
    "uae": "United Arab Emirates", "united arab emirates": "United Arab Emirates",
    "the gambia": "Gambia", "gambia": "Gambia",
    "curacao": "Curaçao", "curaçao": "Curaçao",
    "st kitts and nevis": "Saint Kitts and Nevis",
    "north macedonia": "North Macedonia", "macedonia": "North Macedonia", "fyr macedonia": "North Macedonia",
}


def norm_name(s):
    """Squash a name down for comparison: no accents, lowercase, letters and digits only."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _alias_key(s):
    """The form the ALIASES keys are written in: no accents, lowercase, single spaces."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip()


def canonical(name):
    """Look a name up in the alias table (ignoring case and accents). Unknown names come back as-is."""
    return ALIASES.get(_alias_key(name), name)


def resolve_name(name, known, known_norm=None):
    """Match a name from outside (user input, an odds feed) to one of the names in `known`.

    Tries, in order: the alias table, an exact match, a match after stripping accents and
    punctuation, a match where one name's words are a subset of the other's ("Ireland" vs
    "Republic of Ireland"), and finally a fuzzy match that has to be at least 90% similar.
    If none of those give a single clear answer, the input is returned untouched.
    """
    name = canonical(name)
    known = list(known)
    if name in known:
        return name
    known_norm = known_norm or {norm_name(k): k for k in known}
    nn = norm_name(name)
    if nn in known_norm:
        return known_norm[nn]
    # one name is the other plus or minus a word; only accept it if exactly one team fits
    q_toks = [norm_name(t) for t in str(name).split() if t]
    if not q_toks:
        return name
    qset = set(q_toks)
    sub = [k for k in known
           if (kt := set(norm_name(t) for t in k.split() if t))
           and (kt <= qset or qset <= kt) and len(kt & qset) >= 1]
    if len(sub) == 1:
        return sub[0]
    # last resort: fuzzy match, kept strict on purpose (a wrong match is worse than no match)
    best, best_s = None, 0.0
    for k in known:
        s = difflib.SequenceMatcher(None, nn, norm_name(k)).ratio()
        if s > best_s:
            best, best_s = k, s
    return best if best_s >= 0.90 else name
