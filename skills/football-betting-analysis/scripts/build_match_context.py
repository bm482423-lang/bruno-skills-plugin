#!/usr/bin/env python3
"""
build_match_context.py
=====================
Pre-match football context builder for FlashScore data.
Receives event_id, home_team_id, away_team_id as arguments,
executes all FlashScore endpoint calls, normalizes the data,
and emits a single `final_context` JSON to stdout.

Usage:
    python build_match_context.py <event_id> <home_team_id> <away_team_id>
"""

import os
import sys
import json
import time
import re
import html
import random
import math
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_MATCHES = 10
MAX_WORKERS = 6
API_SLEEP_INITIAL = 1.0
API_SLEEP_BETWEEN = 0.4

# Football domain constants
PARTY_MINUTES = 90.0          # minutes per match for per-90 calculations
FORM_DIFF_THRESHOLD = 0.5    # goal-average threshold for market incoherence check
FAVOURITE_STRONG_THRESHOLD = 60.0   # market prob (%) for strong favourite
FAVOURITE_LEAN_THRESHOLD = 50.0     # market prob (%) for lean favourite
MAX_AVG_GOALS_H2H = 5        # alert threshold for H2H avg goals

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_rapidapi_key() -> str:
    # 1. Environment variable (works in any agent/environment)
    env_key = os.environ.get("RAPIDAPI_KEY", "").strip()
    if env_key:
        return env_key

    # 2. Claude settings files (default for Claude Code users)
    settings_paths = [
        Path.home() / ".claude" / "settings.json",
        Path.home() / ".claude" / "settings.local.json",
    ]
    for path in settings_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "RAPIDAPI_KEY" in data:
                        return data["RAPIDAPI_KEY"]
            except Exception:
                pass

    # 3. Local .rapidapi_key file in the script's directory (fallback for other agents)
    script_dir = Path(__file__).parent
    local_key_file = script_dir / ".rapidapi_key"
    if local_key_file.exists():
        try:
            return local_key_file.read_text().strip()
        except Exception:
            pass

    return ""

RAPIDAPI_KEY = load_rapidapi_key()
RAPIDAPI_HOST = "flashscore4.p.rapidapi.com"

HEADERS = {
    "Content-Type": "application/json",
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
    "timezone": "America/New_York",
}

BASE_URL = f"https://{RAPIDAPI_HOST}/api/flashscore/v2"

# =============================================================================
# HTTP CLIENT
# =============================================================================

def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Make a GET request to the FlashScore RapidAPI endpoint."""
    url = BASE_URL + path
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[WARNING] API call failed for {path}: {e}", file=sys.stderr)
        return None



# =============================================================================
# FETCH FUNCTIONS — one per endpoint
# =============================================================================

def fetch_match_details(event_id: str) -> Optional[Dict]:
    """Get full event details + tournament/team info + initial odds."""
    return api_get("/matches/details", params={"match_id": event_id})


def fetch_match_odds(event_id: str, geo: str = "US") -> Optional[List]:
    """Get all betting markets/odds for the event."""
    return api_get("/matches/odds", params={"match_id": event_id, "geo_ip_code": geo})


def fetch_match_stats(event_id: str) -> Optional[Dict]:
    """Get match statistics (possession, shots, corners, cards, xG)."""
    return api_get("/matches/match/stats", params={"match_id": event_id})


def fetch_match_player_stats(event_id: str) -> Optional[Dict]:
    """Get per-player statistics for the match."""
    return api_get("/matches/match/player-stats", params={"match_id": event_id})


def fetch_match_lineups(event_id: str) -> Optional[List]:
    """Get lineups + missing players."""
    return api_get("/matches/match/lineups", params={"match_id": event_id})


def fetch_match_summary(event_id: str) -> Optional[List]:
    """Get match summary with key events (goals, cards, etc.)."""
    return api_get("/matches/match/summary", params={"match_id": event_id})


def fetch_match_commentary(event_id: str) -> Optional[List]:
    """Get minute-by-minute commentary."""
    return api_get("/matches/match/commentary", params={"match_id": event_id})

def fetch_team_results(team_id: str, page: int = 1) -> Optional[Dict]:
    """Get recent match results for a team."""
    return api_get("/teams/results", params={"team_id": team_id, "page": page})


def build_preview_slug(team_url: str) -> str:
    """Convert team URL to URL slug format for FlashScore preview URL."""
    if not team_url: return ""
    slug = team_url.split("/")[2]
    slug = re.sub(r"[áàä]", "a", slug)
    slug = re.sub(r"[éèë]", "e", slug)
    slug = re.sub(r"[íìï]", "i", slug)
    slug = re.sub(r"[óòö]", "o", slug)
    slug = re.sub(r"[úùü]", "u", slug)
    slug = slug.replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")

def fix_mojibake(text: str) -> str:
    """
    Fix common mojibake issues like:
    'PeÃ±arol' -> 'Peñarol'
    'verÃ¡n' -> 'verán'
    """
    if not text:
        return text

    # Caso típico: UTF-8 interpretado como latin1
    try:
        repaired = text.encode("latin1").decode("utf-8")
        text = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    replacements = {
        "â€œ": '"',
        "â€\x9d": '"',
        "â€˜": "'",
        "â€™": "'",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "Â ": " ",
        "Â": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def clean_preview(preview: Optional[str]) -> Optional[str]:
    """
    Clean Flashscore preview text:
    - fixes encoding issues
    - removes pseudo-tags
    - removes sponsored section
    - removes line breaks (returns single paragraph)
    """
    if not preview:
        return None

    text = html.unescape(preview)
    text = text.replace("\\/", "/")
    text = fix_mojibake(text)

    # Remove sponsored section first (before tag replacements)
    text = re.split(r"Patrocinado:", text, flags=re.IGNORECASE)[0]

    # Structural replacements: close tag -> separator, open tag -> delete
    TAG_REPLACEMENTS = {
        "[/h2]": ". ", "[/p]": " ", "[/b]": "", "[/a]": "",
        "[h2]": "", "[p]": "", "[b]": "", "[a]": "",
    }
    for tag, replacement in TAG_REPLACEMENTS.items():
        text = text.replace(tag, replacement)

    # Remove remaining opening pseudo-tags like [a ...]
    text = re.sub(r"\[a[^\]]*\]", "", text)

    # Normalize whitespace
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def extract_preview_from_dom(resp_text: str) -> Optional[str]:
    """
    Try to extract preview from rendered HTML block.
    """
    soup = BeautifulSoup(resp_text, "html.parser")

    selectors = [
        "div.section--preview div.fp-body_9caht",
        "div.section--preview div.preview__block",
        "div.loadable.complete.section.section--preview div.preview__block",
    ]

    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = element.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text

    return None


def extract_preview_from_content_parsed(resp_text: str) -> Optional[str]:
    """
    Extract preview from embedded eventPreview.contentParsed JSON-like string.
    """
    patterns = [
        r'"eventPreview":\{.*?"contentParsed":"(.*?)","editedAt":',
        r'"contentParsed":"(.*?)","editedAt":',
    ]

    for pattern in patterns:
        match = re.search(pattern, resp_text, re.DOTALL)
        if match:
            return match.group(1)

    return None


def fetch_preview(home_slug: Dict, away_slug: Dict, event_id: str) -> Optional[str]:
    """
    Scrape and clean preview text from FlashScore page.
    Tries both URL orders, DOM extraction first, then embedded contentParsed fallback.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
    }

    candidate_urls = [
        (
            f"https://www.flashscore.co/partido/futbol/"
            f"{home_slug['slug']}-{home_slug['id']}/"
            f"{away_slug['slug']}-{away_slug['id']}/?mid={event_id}"
        ),
        (
            f"https://www.flashscore.co/partido/futbol/"
            f"{away_slug['slug']}-{away_slug['id']}/"
            f"{home_slug['slug']}-{home_slug['id']}/?mid={event_id}"
        ),
    ]

    for url in candidate_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            # 1) Try DOM
            preview = extract_preview_from_dom(resp.text)
            if preview:
                cleaned = clean_preview(preview)
                if cleaned:
                    return cleaned

            # 2) Try embedded eventPreview.contentParsed
            preview = extract_preview_from_content_parsed(resp.text)
            if preview:
                cleaned = clean_preview(preview)
                if cleaned:
                    return cleaned

        except Exception as e:
            print(f"[WARNING] Preview scrape failed for {event_id} in {url}: {e}", file=sys.stderr)

    return None

def fetch_tournament_standings(tournament_id: str, tournament_stage_id: str, stype: str = "overall") -> Optional[List]:
    """Get tournament standings (overall/home/away)."""
    return api_get(
        "/tournaments/standings",
        params={
            "tournament_id": tournament_id,
            "tournament_stage_id": tournament_stage_id,
            "type": stype,
        },
    )


def fetch_match_standings_form(event_id: str, stype: str = "overall") -> Optional[List]:
    """Get recent form standings for teams in the match context."""
    return api_get(
        "/matches/standings/form",
        params={"match_id": event_id, "type": stype},
    )


def fetch_match_standings_overunder(event_id: str, sub_type: str = "2.5", stype: str = "overall") -> Optional[List]:
    """Get Over/Under standings for the match context."""
    return api_get(
        "/matches/standings/over-under",
        params={"match_id": event_id, "type": stype, "sub_type": sub_type},
    )


def fetch_match_top_scorers(event_id: str) -> Optional[List]:
    """Get top scorers related to the match."""
    return api_get("/matches/standings/top-scorers", params={"match_id": event_id})


def fetch_tournament_top_scorers(tournament_id: str, tournament_stage_id: str) -> Optional[List]:
    """Get tournament top scorers."""
    return api_get(
        "/tournaments/standings/top-scorers",
        params={"tournament_id": tournament_id, "tournament_stage_id": tournament_stage_id},
    )


# =============================================================================
# NORMALIZE FUNCTIONS
# =============================================================================

def _extract_1x2_odds(market_group: Dict, home_epid: str, away_epid: str) -> tuple:
    """Extract 1X2 odds. Returns (home, draw, away) odds."""
    home_odd, draw_odd, away_odd = None, None, None
    for odd in market_group.get("odds", []):
        epid = odd.get("eventParticipantId")
        val = parse_odd(odd.get("value"))
        if epid == home_epid:
            home_odd = best_odd(home_odd, val)
        elif epid == away_epid:
            away_odd = best_odd(away_odd, val)
        elif epid is None:
            draw_odd = best_odd(draw_odd, val)
    return home_odd, draw_odd, away_odd


def _extract_overunder_odds(market_group: Dict) -> tuple:
    """Extract Over/Under 2.5 odds. Returns (over_25, under_25)."""
    over_odd, under_odd = None, None
    for odd in market_group.get("odds", []):
        handicap = odd.get("handicap") or {}
        line = str(handicap.get("value", ""))
        sel = (odd.get("selection") or "").upper()
        val = parse_odd(odd.get("value"))
        if line == "2.5":
            if sel == "OVER":
                over_odd = best_odd(over_odd, val)
            elif sel == "UNDER":
                under_odd = best_odd(under_odd, val)
    return over_odd, under_odd


def _extract_btts_odds(market_group: Dict) -> tuple:
    """Extract BTTS odds. Returns (btts_yes, btts_no)."""
    yes_odd, no_odd = None, None
    for odd in market_group.get("odds", []):
        btts = odd.get("bothTeamsToScore")
        val = parse_odd(odd.get("value"))
        if btts is True:
            yes_odd = best_odd(yes_odd, val)
        elif btts is False:
            no_odd = best_odd(no_odd, val)
    return yes_odd, no_odd


def normalize_odds(odds_data: List, home_epid: str, away_epid: str) -> Dict:
    """
    Extract and normalize the most relevant odds markets.
    Returns available markets + best odds for 1X2, Over/Under, BTTS.
    """
    result = {
        "available_markets": [],
        "odds_home": None,
        "odds_draw": None,
        "odds_away": None,
        "odds_over_25": None,
        "odds_under_25": None,
        "odds_btts_yes": None,
        "odds_btts_no": None,
        "warnings": [],
    }

    if not odds_data:
        result["warnings"].append("No odds data available [N/A]")
        return result

    all_markets = set()

    for bookmaker in odds_data:
        for market_group in bookmaker.get("odds", []):
            betting_type = market_group.get("bettingType", "")
            scope = market_group.get("bettingScope", "")
            key = f"{scope}_{betting_type}"
            all_markets.add(key)

            if betting_type == "HOME_DRAW_AWAY" and scope == "FULL_TIME":
                h, d, a = _extract_1x2_odds(market_group, home_epid, away_epid)
                result["odds_home"] = best_odd(result["odds_home"], h)
                result["odds_draw"] = best_odd(result["odds_draw"], d)
                result["odds_away"] = best_odd(result["odds_away"], a)

            elif betting_type == "OVER_UNDER" and scope == "FULL_TIME":
                o, u = _extract_overunder_odds(market_group)
                result["odds_over_25"] = best_odd(result["odds_over_25"], o)
                result["odds_under_25"] = best_odd(result["odds_under_25"], u)

            elif betting_type == "BOTH_TEAMS_TO_SCORE" and scope == "FULL_TIME":
                y, n = _extract_btts_odds(market_group)
                result["odds_btts_yes"] = best_odd(result["odds_btts_yes"], y)
                result["odds_btts_no"] = best_odd(result["odds_btts_no"], n)

    result["available_markets"] = sorted(list(all_markets))
    return result


def parse_odd(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def best_odd(current: Optional[float], new: Optional[float]) -> Optional[float]:
    """Prefer higher odds (best market value)."""
    if current is None:
        return new
    if new is None:
        return current
    return max(current, new)


def normalize_implied_probs(odds_data: Dict) -> Dict:
    """Calculate implied probabilities from normalized odds."""
    probs = {}
    for market, odd_key in [
        ("prob_home", "odds_home"),
        ("prob_draw", "odds_draw"),
        ("prob_away", "odds_away"),
        ("prob_over_25", "odds_over_25"),
        ("prob_btts_yes", "odds_btts_yes"),
    ]:
        odd = odds_data.get(odd_key)
        if odd and odd > 0:
            probs[market] = round((1 / odd) * 100, 2)
        else:
            probs[market] = None
    return probs


def _resolve_goals(m: Dict) -> tuple:
    """
    Extract goals_for, goals_against, total_goals from a match record.
    Supports two formats:
    - h2h: home_score, away_score + team_is_home
    - team results: goals_for, goals_against + total_goals
    Returns (goals_for, goals_against, total_goals).
    """
    if "home_score" in m and "away_score" in m:
        is_home = m.get("team_is_home", True)
        hs = m.get("home_score", 0) or 0
        aw = m.get("away_score", 0) or 0
        return (hs if is_home else aw), (aw if is_home else hs), hs + aw
    return (
        m.get("goals_for", 0) or 0,
        m.get("goals_against", 0) or 0,
        m.get("total_goals", 0),
    )


def _compute_results_basic_stats(matches: List[Dict]) -> Dict:
    """
    Compute basic stats (all_matches, form_string, points, home_gf_avg,
    home_gc_avg, over_25_freq, btts_freq, home_*/away_*) from a list of matches.

    Renamed fields for disambiguation in h2h context:
      gf_avg     -> home_gf_avg   (analyzed team goals scored avg when playing at home)
      gc_avg     -> home_gc_avg   (analyzed team goals conceded avg when playing at home)
      away_gf_avg -> away_team_gf_avg
      away_gc_avg -> away_team_gc_avg
    """
    n = len(matches)
    if n == 0:
        return {
            "all_matches": 0, "form_string": None, "points": 0,
            "home_gf_avg": None, "home_gc_avg": None,
            "over_25_freq": None, "btts_freq": None,
        }

    form_records, pts = [], 0
    gf, gc = 0, 0
    over_count, btts_count = 0, 0

    for m in matches:
        goals_for, goals_against, total_goals = _resolve_goals(m)
        gf += goals_for
        gc += goals_against
        if total_goals > 2.5:
            over_count += 1
        if goals_for > 0 and goals_against > 0:
            btts_count += 1
        if goals_for > goals_against:
            form_records.append("W")
            pts += 3
        elif goals_for == goals_against:
            form_records.append("D")
            pts += 1
        else:
            form_records.append("L")

    result = {
        "all_matches": n,
        "form_string": "".join(form_records) or None,
        "points": pts,
        "home_gf_avg": round(gf / n, 2),
        "home_gc_avg": round(gc / n, 2),
        "over_25_freq": f"{over_count}/{n}",
        "btts_freq": f"{btts_count}/{n}",
    }

    home_matches = [m for m in matches if m.get("team_is_home")]
    away_matches = [m for m in matches if not m.get("team_is_home")]

    if home_matches:
        h_gf = sum(_resolve_goals(m)[0] for m in home_matches)
        h_gc = sum(_resolve_goals(m)[1] for m in home_matches)
        h_pts = sum(3 if _resolve_goals(m)[0] > _resolve_goals(m)[1]
                    else 1 if _resolve_goals(m)[0] == _resolve_goals(m)[1]
                    else 0 for m in home_matches)
        k = len(home_matches)
        result["home_ppg"] = round(h_pts / k, 2)
        result["home_gf_avg"] = round(h_gf / k, 2)
        result["home_gc_avg"] = round(h_gc / k, 2)

    if away_matches:
        a_gf = sum(_resolve_goals(m)[0] for m in away_matches)
        a_gc = sum(_resolve_goals(m)[1] for m in away_matches)
        a_pts = sum(3 if _resolve_goals(m)[0] > _resolve_goals(m)[1]
                    else 1 if _resolve_goals(m)[0] == _resolve_goals(m)[1]
                    else 0 for m in away_matches)
        k = len(away_matches)
        result["away_ppg"] = round(a_pts / k, 2)
        result["away_team_gf_avg"] = round(a_gf / k, 2)
        result["away_team_gc_avg"] = round(a_gc / k, 2)

    return result


def _compute_h2h_basic_stats(matches: List[Dict], current_home_name: str, current_away_name: str) -> Dict:
    """
    Compute basic stats for h2h context, where 'home' always refers to the
    team that IS the HOME team in the CURRENT analyzed match (e.g. Bayern),
    and 'away' refers to the team that IS the AWAY team in the CURRENT analyzed
    match (e.g. Real Madrid).

    current_home_name / current_away_name identify which team in each h2h match
    corresponds to current home/away, regardless of whether that team was
    home or away in the historical h2h encounter.

    For each h2h match, goals are attributed to the current-home / current-away
    team by matching the current team names against home_team / away_team in the
    h2h match object.

    Excludes form_string, points, home_ppg, away_ppg — not meaningful in h2h.
    """
    n = len(matches)
    if n == 0:
        return {
            "home_team": {
                "wins": 0, "losses": 0, "draws": 0,
                "gf_avg": None, "gc_avg": None, "total_goals": 0,
            },
            "away_team": {
                "wins": 0, "losses": 0, "draws": 0,
                "gf_avg": None, "gc_avg": None, "total_goals": 0,
            },
            "both_teams_scored": None,
            "over_25_freq": None,
            "btts_freq": None,
            "total_matches": 0,
            "total_goals": 0,
        }

    home_gf = home_gc = away_gf = away_gc = 0
    home_wins = away_wins = draws = 0
    total_goals = 0
    over_count = btts_count = 0

    for m in matches:
        hs = m.get("home_score", 0) or 0
        aw = m.get("away_score", 0) or 0
        tg = hs + aw

        h2h_home = m.get("home_team", "")
        h2h_away = m.get("away_team", "")

        if h2h_home == current_home_name:
            # Current home team was home in this h2h → their goals are hs
            home_gf += hs
            home_gc += aw
            # Current away team was away in this h2h → their goals are aw
            away_gf += aw
            away_gc += hs
            if hs > aw:
                home_wins += 1
            elif hs < aw:
                away_wins += 1
            else:
                draws += 1
        elif h2h_away == current_home_name:
            # Current home team was away in this h2h → their goals are aw
            home_gf += aw
            home_gc += hs
            # Current away team was home in this h2h → their goals are hs
            away_gf += hs
            away_gc += aw
            if aw > hs:
                home_wins += 1
            elif aw < hs:
                away_wins += 1
            else:
                draws += 1

        total_goals += tg
        if tg > 2.5:
            over_count += 1
        if hs > 0 and aw > 0:
            btts_count += 1

    result = {
        "home_team": {
            "wins": home_wins,
            "losses": away_wins,
            "draws": draws,
            "gf_avg": round(home_gf / n, 2),
            "gc_avg": round(home_gc / n, 2),
            "total_goals": home_gf,
        },
        "away_team": {
            "wins": away_wins,
            "losses": home_wins,
            "draws": draws,
            "gf_avg": round(away_gf / n, 2),
            "gc_avg": round(away_gc / n, 2),
            "total_goals": away_gf,
        },
        "both_teams_scored": f"{btts_count}/{n}",
        "over_25_freq": f"{over_count}/{n}",
        "btts_freq": f"{btts_count}/{n}",
        "total_matches": n,
        "total_goals": total_goals,
    }

    return result


def build_h2h_from_results(home_team_results: Dict, away_team_results: Dict, home_name: str, away_name: str) -> Dict:
    """
    Build H2H matches from the two teams' match histories.
    Filters matches where the opponent was the direct rival.
    This replaces the separate H2H API call to avoid redundancy.
    """
    match_map: Dict[str, Dict] = {}
    home_matches = home_team_results.get("matches", [])
    away_matches = away_team_results.get("matches", [])

    def to_actual_scores(m: Dict) -> tuple[int, int]:
        """
        Convert team-perspective goals_for/goals_against into actual
        home_score and away_score based on team_is_home.
        """
        gf = m.get("goals_for", 0)
        ga = m.get("goals_against", 0)
        is_home = m.get("team_is_home")

        if is_home is True:
            return gf, ga
        else:
            return ga, gf

    def to_actual_teams(m: Dict, team_name: str) -> tuple[str, str]:
        """
        Resolve actual home_team and away_team names using team_is_home.
        """
        opponent = m.get("opponent", "")
        is_home = m.get("team_is_home")

        if is_home is True:
            return team_name, opponent
        else:
            return opponent, team_name

    # From home team's history
    for m in home_matches:
        if m.get("opponent") == away_name:
            mid = m.get("match_id")
            home_score, away_score = to_actual_scores(m)
            actual_home_team, actual_away_team = to_actual_teams(m, home_name)
            is_home = m.get("team_is_home", True)

            match_map[mid] = {
                "match_id": mid,
                "timestamp": m.get("timestamp"),
                "home_score": home_score,
                "away_score": away_score,
                "home_team": actual_home_team,
                "away_team": actual_away_team,
                "tournament_id": m.get("tournament_id", ""),
                "tournament_name": m.get("tournament_name", ""),
                "team_is_home": is_home,
            }

    # From away team's history
    for m in away_matches:
        if m.get("opponent") == home_name:
            mid = m.get("match_id")
            if mid in match_map:
                continue

            home_score, away_score = to_actual_scores(m)
            actual_home_team, actual_away_team = to_actual_teams(m, away_name)
            is_home = False  # away team in analysis was away in this match

            match_map[mid] = {
                "match_id": mid,
                "timestamp": m.get("timestamp"),
                "home_score": home_score,
                "away_score": away_score,
                "home_team": actual_home_team,
                "away_team": actual_away_team,
                "tournament_id": m.get("tournament_id", ""),
                "tournament_name": m.get("tournament_name", ""),
                "team_is_home": is_home,
            }

    matches = sorted(match_map.values(), key=lambda r: r.get("timestamp") or 0, reverse=True)

    # Compute h2h basic stats with explicit team breakdown
    form_basic = _compute_h2h_basic_stats(matches, home_name, away_name)

    return {
        "matches": matches,
        "basic_stats": form_basic,
        "warnings": [],
    }


def normalize_team_results(team_results_data: Dict, team_name: str, team_id: str) -> Dict:
    """
    Normalize a team's match history.
    Compute form (W/D/L), points, GF/GA averages, Over 2.5/BTTS frequency.
    """
    if not team_results_data:
        return {"matches": [], "form": None, "warnings": [f"Team results not available for {team_name} [N/A]"]}

    leagues = team_results_data if isinstance(team_results_data, list) else [team_results_data]
    all_matches = []

    for league_block in leagues:
        for league in league_block.get("leagues", []) if "leagues" in league_block else [league_block]:
            tournament_id = league.get("tournament_id")
            tournament_name = league.get("full_name")
            
            for match in league.get("matches", []):
                try:
                    ts = match.get("timestamp")
                    home_t = match.get("home_team", {})
                    away_t = match.get("away_team", {})
                    scores = match.get("scores", {})

                    h_score = int(scores.get("home", 0) or 0)
                    a_score = int(scores.get("away", 0) or 0)

                    # The API returns the team in the "away_team" field of the response,
                    # regardless of whether they were home or away in the actual match.
                    # We determine their real role by checking which field contains them.
                    team_in_home = home_t.get("team_id") == team_id
                    team_in_away = away_t.get("team_id") == team_id

                    if team_in_home and not team_in_away:
                        # Team was actually home in this match
                        is_home = True
                        opponent_name = away_t.get("name")
                        goals_for = h_score
                        goals_against = a_score
                    elif team_in_away and not team_in_home:
                        # Team was actually away in this match
                        is_home = False
                        opponent_name = home_t.get("name")
                        goals_for = a_score
                        goals_against = h_score
                    else:
                        # Team not found in either position — skip
                        continue

                    all_matches.append({
                        "tournament_id": tournament_id,
                        "tournament_name": tournament_name,
                        "match_id": match.get("match_id"),
                        "timestamp": ts,
                        "team_is_home": is_home,
                        "opponent": opponent_name,
                        "goals_for": goals_for,
                        "goals_against": goals_against,
                        "total_goals": h_score + a_score,
                        "both_teams_scored": h_score > 0 and a_score > 0,
                    })
                except (TypeError, ValueError):
                    continue

    # Sort by timestamp descending (most recent first)
    all_matches.sort(key=lambda m: m.get("timestamp") or 0, reverse=True)
    all_matches = all_matches[:MAX_MATCHES]

    # Compute basic stats via shared helper
    basic_stats = _compute_results_basic_stats(all_matches)

    # Add explicit top-level summary stats into basic_stats
    explicit = _compute_top_level_stats(all_matches)
    basic_stats.update({
        "total_matches": explicit["total_matches"],
        "wins": explicit["wins"],
        "draws": explicit["draws"],
        "losses": explicit["losses"],
        "goals_for": explicit["goals_for"],
        "goals_against": explicit["goals_against"],
        "total_goals": explicit["total_goals"],
        "both_teams_scored": explicit["both_teams_scored"],
    })

    result = {
        "matches": all_matches,
        "basic_stats": basic_stats,
        "warnings": [],
    }

    return result


# =============================================================================
# ADVANCED STATS NORMALIZATION
# =============================================================================

STAT_NAME_TO_KEY = {
    "Expected goals (xG)": "xg",
    "Goals": "goals",
    "Ball possession": "possession",
    "Total shots": "shots",
    "Shots on target": "shots_on_target",
    "Shots off target": "shots_off_target",
    "Blocked shots": "blocked_shots",
    "Shots inside the box": "shots_inside_box",
    "Shots outside the box": "shots_outside_box",
    "Big chances": "big_chances",
    "Corner kicks": "corners",
    "Touches in opposition box": "touches_in_opposition_box",
    "Hit the woodwork": "hit_woodwork",
    "Accurate through passes": "accurate_through_passes",
    "Offsides": "offsides",
    "Free kicks": "free_kicks",
    "Throw ins": "throw_ins",
    "Passes": "passes",
    "Long passes": "long_passes",
    "Passes in final third": "passes_final_third",
    "Crosses": "crosses",
    "Expected assists (xA)": "xa",
    "xG on target (xGOT)": "xgot",
    "Headed goals": "headed_goals",
    "Fouls": "fouls",
    "Tackles": "tackles",
    "Duels won": "duels_won",
    "Clearances": "clearances",
    "Interceptions": "interceptions",
    "Errors leading to shot": "errors_leading_to_shot",
    "Errors leading to goal": "errors_leading_to_goal",
    "Goalkeeper saves": "goalkeeper_saves",
    "xGOT faced": "xgot_faced",
    "Goals prevented": "goals_prevented",
    "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
}

PASS_TYPE_STATS = {"passes", "long_passes", "passes_final_third", "crosses", "tackles"}


def stat_name_to_key(name: str) -> Optional[str]:
    """Map API stat name to normalized key."""
    return STAT_NAME_TO_KEY.get(name)


def parse_stat_value(name: str, home_val: Any, away_val: Any) -> Dict:
    """
    Parse raw stat values into {for, against} structure.
    Uses parse_pass_stat for pass-type stats, extracts possession percentage,
    and direct value extraction for everything else.
    """
    key = stat_name_to_key(name)

    if key in PASS_TYPE_STATS:
        home_parsed = parse_pass_stat(home_val)
        away_parsed = parse_pass_stat(away_val)
        return {
            "for": home_parsed,
            "against": away_parsed,
        }

    if key == "possession":
        home_int = int(re.sub(r"\D", "", str(home_val))) if home_val else 0
        away_int = int(re.sub(r"\D", "", str(away_val))) if away_val else 0
        # Ensure for+against sums to 100
        return {"for": home_int, "against": away_int}

    def direct(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        s = str(val).strip()
        if s == "-" or s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return s

    return {"for": direct(home_val), "against": direct(away_val)}


def normalize_advanced_stats(raw_stats: Optional[Dict], *, team_is_home: bool = True) -> Dict:
    """
    Normalize raw output from fetch_match_stats() into structured advanced_stats.
    Handles local/away orientation: the API always returns home_team/away_team from
    the perspective of who was home in that specific match. When team_is_home=False
    we swap "for" and "against" to express stats from the visitor's perspective.

    Returns:
        {
          "match":     { stat_key: { "for": value, "against": value }, ... },
          "1st-half":  { stat_key: { "for": value, "against": value }, ... },
          "2nd-half":  { stat_key: { "for": value, "against": value }, ... },
          "warnings":  []
        }
    """
    result = {
        "match": {},
        "1st-half": {},
        "2nd-half": {},
        "warnings": [],
    }

    if not raw_stats:
        return result

    for period_key in ("match", "1st-half", "2nd-half"):
        period_data = raw_stats.get(period_key, [])
        if not isinstance(period_data, list):
            continue

        seen_names: set = set()

        for item in period_data:
            if not isinstance(item, dict):
                continue

            stat_name = item.get("name")
            if not stat_name:
                continue

            # Deduplicate by stat name within each period
            if stat_name in seen_names:
                continue
            seen_names.add(stat_name)

            key = stat_name_to_key(stat_name)
            if key is None:
                continue

            home_val = item.get("home_team")
            away_val = item.get("away_team")
            parsed = parse_stat_value(stat_name, home_val, away_val)
            # API returns home_team/away_team from the perspective of who was home
            # in that match. When team_is_home=False (we were the away team), swap
            # "for" and "against" so stats are always expressed from our team's view.
            if not team_is_home:
                parsed = {"for": parsed.get("against"), "against": parsed.get("for")}
            result[period_key][key] = parsed

    return result


# =============================================================================
# =============================================================================
# ADVANCED FORM AGGREGATION
# =============================================================================

def _avg_stat(matches, period, key, side, sub_key=None):
    vals = []
    for m in matches:
        stats = m.get('advanced_stats', {}).get(period, {})
        val = stats.get(key, {}).get(side)
        if sub_key and isinstance(val, dict):
            val = val.get(sub_key)
        if val is not None:
            vals.append(val)
    return round(sum(vals) / len(vals), 2) if vals else None


def _avg_goals_from_result(matches, side):
    key = 'goals_for' if side == 'for' else 'goals_against'
    vals = [m.get(key) for m in matches if m.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _avg_bc_conversion(matches, period):
    vals = []
    for m in matches:
        bc = m.get('advanced_stats', {}).get(period, {}).get('big_chances', {}).get('for')
        gf = m.get('goals_for')
        if bc is not None and bc > 0 and gf is not None:
            vals.append(gf / bc)
    return round(sum(vals) / len(vals) * 100, 2) if vals else None


def _safe(val, default=0.0):
    return val if val is not None else default


def _category_attack(matches, period):
    return {
        'goals_for_avg': _avg_goals_from_result(matches, 'for'),
        'xg_for_avg': _avg_stat(matches, period, 'xg', 'for'),
        'xgot_for_avg': _avg_stat(matches, period, 'xgot', 'for'),
        'xa_for_avg': _avg_stat(matches, period, 'xa', 'for'),
        'shots_for_avg': _avg_stat(matches, period, 'shots', 'for'),
        'shots_on_target_for_avg': _avg_stat(matches, period, 'shots_on_target', 'for'),
        'shots_off_target_for_avg': _avg_stat(matches, period, 'shots_off_target', 'for'),
        'blocked_shots_for_avg': _avg_stat(matches, period, 'blocked_shots', 'for'),
        'shots_inside_box_for_avg': _avg_stat(matches, period, 'shots_inside_box', 'for'),
        'shots_outside_box_for_avg': _avg_stat(matches, period, 'shots_outside_box', 'for'),
        'big_chances_for_avg': _avg_stat(matches, period, 'big_chances', 'for'),
        'touches_in_opposition_box_avg': _avg_stat(matches, period, 'touches_in_opposition_box', 'for'),
        'hit_woodwork_avg': _avg_stat(matches, period, 'hit_woodwork', 'for'),
    }


def _category_defense(matches, period):
    return {
        'goals_against_avg': _avg_goals_from_result(matches, 'against'),
        'xg_against_avg': _avg_stat(matches, period, 'xg', 'against'),
        'xgot_faced_avg': _avg_stat(matches, period, 'xgot_faced', 'for'),
        'shots_against_avg': _avg_stat(matches, period, 'shots', 'against'),
        'shots_on_target_against_avg': _avg_stat(matches, period, 'shots_on_target', 'against'),
        'shots_off_target_against_avg': _avg_stat(matches, period, 'shots_off_target', 'against'),
        'blocked_shots_against_avg': _avg_stat(matches, period, 'blocked_shots', 'against'),
        'shots_inside_box_against_avg': _avg_stat(matches, period, 'shots_inside_box', 'against'),
        'shots_outside_box_against_avg': _avg_stat(matches, period, 'shots_outside_box', 'against'),
        'big_chances_against_avg': _avg_stat(matches, period, 'big_chances', 'against'),
        'touches_in_opposition_box_against_avg': _avg_stat(matches, period, 'touches_in_opposition_box', 'against'),
        'goalkeeper_saves_avg': _avg_stat(matches, period, 'goalkeeper_saves', 'for'),
        'errors_leading_to_shot_avg': _avg_stat(matches, period, 'errors_leading_to_shot', 'for'),
        'errors_leading_to_goal_avg': _avg_stat(matches, period, 'errors_leading_to_goal', 'for'),
        'goals_prevented_avg': _avg_stat(matches, period, 'goals_prevented', 'for'),
    }


def _category_control(matches, period):
    return {
        'possession_avg': _avg_stat(matches, period, 'possession', 'for'),
        'passes_accuracy_avg': _avg_stat(matches, period, 'passes', 'for', 'pct'),
        'passes_completed_avg': _avg_stat(matches, period, 'passes', 'for', 'completed'),
        'passes_attempted_avg': _avg_stat(matches, period, 'passes', 'for', 'attempted'),
        'long_pass_accuracy_avg': _avg_stat(matches, period, 'long_passes', 'for', 'pct'),
        'long_passes_completed_avg': _avg_stat(matches, period, 'long_passes', 'for', 'completed'),
        'long_passes_attempted_avg': _avg_stat(matches, period, 'long_passes', 'for', 'attempted'),
        'final_third_pass_accuracy_avg': _avg_stat(matches, period, 'passes_final_third', 'for', 'pct'),
        'final_third_passes_completed_avg': _avg_stat(matches, period, 'passes_final_third', 'for', 'completed'),
        'final_third_passes_attempted_avg': _avg_stat(matches, period, 'passes_final_third', 'for', 'attempted'),
        'accurate_through_passes_avg': _avg_stat(matches, period, 'accurate_through_passes', 'for'),
    }


def _category_set_pieces(matches, period):
    return {
        'corners_for_avg': _avg_stat(matches, period, 'corners', 'for'),
        'corners_against_avg': _avg_stat(matches, period, 'corners', 'against'),
        'offsides_for_avg': _avg_stat(matches, period, 'offsides', 'for'),
        'offsides_against_avg': _avg_stat(matches, period, 'offsides', 'against'),
        'free_kicks_for_avg': _avg_stat(matches, period, 'free_kicks', 'for'),
        'free_kicks_against_avg': _avg_stat(matches, period, 'free_kicks', 'against'),
        'throw_ins_for_avg': _avg_stat(matches, period, 'throw_ins', 'for'),
        'throw_ins_against_avg': _avg_stat(matches, period, 'throw_ins', 'against'),
        'cross_accuracy_avg': _avg_stat(matches, period, 'crosses', 'for', 'pct'),
        'crosses_completed_avg': _avg_stat(matches, period, 'crosses', 'for', 'completed'),
        'crosses_attempted_avg': _avg_stat(matches, period, 'crosses', 'for', 'attempted'),
    }


def _category_discipline(matches, period):
    yf = _avg_stat(matches, period, 'yellow_cards', 'for')
    ya = _avg_stat(matches, period, 'yellow_cards', 'against')
    rf = _avg_stat(matches, period, 'red_cards', 'for')
    if rf is None and yf is not None:
        rf = 0.0
    if yf is not None and rf is not None:
        cards_total = round(yf + rf, 2)
    elif yf is not None:
        cards_total = yf
    elif rf is not None:
        cards_total = rf
    else:
        cards_total = 0.0
    return {
        'yellow_cards_avg': yf,
        'red_cards_avg': rf,
        'cards_total_avg': cards_total,
        'fouls_committed_avg': _avg_stat(matches, period, 'fouls', 'for'),
    }


def _category_duels(matches, period):
    return {
        'tackles_success_pct_avg': _avg_stat(matches, period, 'tackles', 'for', 'pct'),
        'tackles_won_avg': _avg_stat(matches, period, 'tackles', 'for', 'completed'),
        'tackles_attempted_avg': _avg_stat(matches, period, 'tackles', 'for', 'attempted'),
        'duels_won_avg': _avg_stat(matches, period, 'duels_won', 'for'),
        'clearances_avg': _avg_stat(matches, period, 'clearances', 'for'),
        'interceptions_avg': _avg_stat(matches, period, 'interceptions', 'for'),
    }


def _category_efficiency(matches, period):
    sf = _avg_stat(matches, period, 'shots', 'for')
    sot_f = _avg_stat(matches, period, 'shots_on_target', 'for')
    sot_a = _avg_stat(matches, period, 'shots_on_target', 'against')
    xgf = _avg_stat(matches, period, 'xg', 'for')
    xga = _avg_stat(matches, period, 'xg', 'against')
    bcf = _avg_stat(matches, period, 'big_chances', 'for')
    sv_f = _avg_stat(matches, period, 'goalkeeper_saves', 'for')
    gf = _avg_goals_from_result(matches, 'for')
    ga = _avg_goals_from_result(matches, 'against')

    def pct(a, b):
        return round(a / b * 100, 2) if a is not None and b and b > 0 else None

    return {
        'shot_accuracy_pct': pct(sot_f, sf),
        'goal_conversion_pct': min(pct(gf, sf), 100.0) if gf is not None and sf and sf > 0 else None,
        'big_chance_conversion_pct': _avg_bc_conversion(matches, period),
        'xg_per_shot': round(xgf / sf, 2) if xgf is not None and sf and sf > 0 else None,
        'shots_on_target_faced_per_goal_against': (None if ga == 0 else round(sot_a / ga, 2)) if sot_a is not None and ga is not None else None,
        'save_pct': pct(sv_f, sot_a),
        'finishing_overperformance': round(gf - xgf, 2) if gf is not None and xgf is not None else None,
        'conceding_overperformance': round(ga - xga, 2) if ga is not None and xga is not None else None,
    }


def _category_derived(matches, period):
    xgf = _avg_stat(matches, period, 'xg', 'for')
    xga = _avg_stat(matches, period, 'xg', 'against')
    sf = _avg_stat(matches, period, 'shots', 'for')
    sot_f = _avg_stat(matches, period, 'shots_on_target', 'for')
    sot_a = _avg_stat(matches, period, 'shots_on_target', 'against')
    bcf = _avg_stat(matches, period, 'big_chances', 'for')
    bca = _avg_stat(matches, period, 'big_chances', 'against')
    cf = _avg_stat(matches, period, 'corners', 'for')
    ca = _avg_stat(matches, period, 'corners', 'against')
    ycf = _avg_stat(matches, period, 'yellow_cards', 'for')
    yca = _avg_stat(matches, period, 'yellow_cards', 'against')
    s = _safe
    return {
        'xg_balance_avg': round(s(xgf) - s(xga), 2),
        'xg_ratio': round(xgf / s(xga), 2) if xga and xga > 0 else None,
        'shots_share': round(sf / (sf + s(sot_a)), 2) if sf and sf > 0 else None,
        'shots_on_target_share': round(sot_f / (sot_f + s(sot_a)), 2) if sot_f and sot_f > 0 else None,
        'big_chances_balance_avg': round(s(bcf) - s(bca), 2) if bcf is not None else None,
        'corners_balance_avg': round(s(cf) - s(ca), 2) if cf is not None else None,
        'discipline_balance_avg': round(s(ycf) - s(yca), 2) if ycf is not None else None,
    }


def compute_category_averages(matches, period):
    return {
        'attack': _category_attack(matches, period),
        'defense': _category_defense(matches, period),
        'control': _category_control(matches, period),
        'set_pieces_and_territory': _category_set_pieces(matches, period),
        'discipline': _category_discipline(matches, period),
        'duels_and_defending': _category_duels(matches, period),
        'efficiency': _category_efficiency(matches, period),
        'derived': _category_derived(matches, period),
    }


def compute_advanced_form(matches: List[Dict]) -> Dict:
    """Aggregate advanced_stats from matches into organized categories."""
    # Only exclude matches that truly lack advanced_stats (not just because they have warnings)
    valid = [m for m in matches if m.get("advanced_stats")]
    n = len(valid)
    total = len(matches)

    result = {
        "overall": compute_category_averages(valid, "match"),
        "first_half": compute_category_averages(valid, "1st-half"),
        "second_half": compute_category_averages(valid, "2nd-half"),
    }
    if n < total:
        result["warnings"] = [f"advanced_stats_partial: {n}/{total} matches con stats"]
    if n == 0:
        result = {
            "overall": {},
            "first_half": {},
            "second_half": {},
        }
    return result


def compute_h2h_advanced_stats(
    matches: List[Dict],
    stats_map: Dict,
    current_home_name: str,
    current_away_name: str,
) -> Dict:
    """
    Compute advanced stats for h2h context split into two team perspectives:
      - home_team: stats of the team that IS home in the CURRENT match
      - away_team: stats of the team that IS away in the CURRENT match

    For each h2h match we look up raw_stats from stats_map and determine whether
    the current home team was home or away in that historical encounter by matching
    current_home_name against home_team/away_team in the match record. Then we call
    normalize_advanced_stats with the appropriate team_is_home flag so that the
    resulting stats are always expressed from the current team's perspective.

    Each sub-dict has the same structure as compute_advanced_form output:
    { overall: { attack, defense, control, ... }, first_half: {...}, second_half: {...} }
    """
    home_perspective_matches: List[Dict] = []
    away_perspective_matches: List[Dict] = []

    for m in matches:
        mid = m.get("match_id")
        raw_stats = stats_map.get(mid)
        if not raw_stats:
            placeholder = {"match_id": mid, "advanced_stats": {"warnings": ["Stats not available"]}}
            home_perspective_matches.append(placeholder)
            away_perspective_matches.append(placeholder)
            continue

        h2h_home = m.get("home_team", "")
        h2h_away = m.get("away_team", "")

        # Determine if current home team was home or away in this h2h match
        current_home_was_home = (h2h_home == current_home_name)

        if current_home_was_home:
            # Current home team (Bayern) was home in this h2h → their perspective = team_is_home=True
            home_stats = normalize_advanced_stats(raw_stats, team_is_home=True)
            # Current away team (Real) was away in this h2h → their perspective = team_is_home=False
            away_stats = normalize_advanced_stats(raw_stats, team_is_home=False)
        else:
            # Current home team (Bayern) was away in this h2h → their perspective = team_is_home=False
            home_stats = normalize_advanced_stats(raw_stats, team_is_home=False)
            # Current away team (Real) was home in this h2h → their perspective = team_is_home=True
            away_stats = normalize_advanced_stats(raw_stats, team_is_home=True)

        home_perspective_matches.append({"match_id": mid, "advanced_stats": home_stats})
        away_perspective_matches.append({"match_id": mid, "advanced_stats": away_stats})

    return {
        "home_team": compute_advanced_form(home_perspective_matches),
        "away_team": compute_advanced_form(away_perspective_matches),
    }


def normalize_summary(raw: Any) -> Optional[Dict]:
    """
    Normalize match summary (key events: goals, cards, substitutions, etc.).
    Returns {events: [{minutes, team, type, description, players}]}.
    """
    if not raw:
        return None

    events = []
    items = raw if isinstance(raw, list) else raw.get("events", []) if isinstance(raw, dict) else []

    for item in items:
        if not isinstance(item, dict):
            continue

        event_type = item.get("type") or ""
        # Normalize type to our vocabulary
        type_map = {
            "goal": "goal",
            "own_goal": "own_goal",
            "penalty_goal": "penalty_goal",
            "penalty_missed": "penalty_missed",
            "substitution": "substitution",
            "var": "var",
            "yellow_card": "yellow_card",
            "red_card": "red_card",
            "second_yellow": "second_yellow",
        }
        normalized_type = type_map.get(event_type.lower(), event_type)

        # Extract team
        team_val = item.get("team")
        if team_val == "home":
            team = "home"
        elif team_val == "away":
            team = "away"
        else:
            team = None

        # Extract players
        players = []
        for p in item.get("players", []) or []:
            players.append({
                "name": p.get("name"),
                "player_id": p.get("player_id"),
            })

        # Extract minute
        minutes_val = item.get("minutes")
        try:
            minutes = int(minutes_val) if minutes_val is not None else None
        except (ValueError, TypeError):
            minutes = None

        events.append({
            "minutes": minutes,
            "team": team,
            "type": normalized_type,
            "description": item.get("description"),
            "players": players,
        })

    return {"events": events}


def normalize_commentary(raw: Any) -> Optional[Dict]:
    """
    Normalize minute-by-minute commentary.
    Returns {commentary: [{minutes, description}]}.
    """
    if not raw:
        return None

    commentary = []
    items = raw if isinstance(raw, list) else raw.get("commentary", []) if isinstance(raw, dict) else []

    for item in items:
        if not isinstance(item, dict):
            continue

        minutes_val = item.get("minutes")
        try:
            minutes = int(minutes_val) if minutes_val is not None else None
        except (ValueError, TypeError):
            minutes = None

        commentary.append({
            "minutes": minutes,
            "description": item.get("description"),
        })

    return {"commentary": commentary}

def normalize_match_stats(raw: Optional[Dict]) -> Dict:
    """
    Normalize raw output from fetch_match_stats() for live match.
    Unlike normalize_advanced_stats, this does NOT swap for/against
    based on team_is_home — returns raw home/away orientation.

    Returns:
        {
          "match":     { stat_key: { "home": value, "away": value }, ... },
          "1st-half":  { stat_key: { "home": value, "away": value }, ... },
          "2nd-half":  { stat_key: { "home": value, "away": value }, ... },
          "warnings":  []
        }
    """
    result = {
        "match": {},
        "1st-half": {},
        "2nd-half": {},
        "warnings": [],
    }

    if not raw:
        result["warnings"].append("Match stats not available [N/A]")
        return result

    for period_key in ("match", "1st-half", "2nd-half"):
        period_data = raw.get(period_key, [])
        if not isinstance(period_data, list):
            continue

        seen_names: set = set()

        for item in period_data:
            if not isinstance(item, dict):
                continue

            stat_name = item.get("name")
            if not stat_name:
                continue

            if stat_name in seen_names:
                continue
            seen_names.add(stat_name)

            key = stat_name_to_key(stat_name)
            if key is None:
                # Try parsing as pass-type stat
                parsed = parse_pass_stat(item.get("home_team"))
                if parsed.get("pct") is not None:
                    result[period_key][stat_name] = {
                        "home": parsed,
                        "away": parse_pass_stat(item.get("away_team")),
                    }
                continue

            home_val = item.get("home_team")
            away_val = item.get("away_team")

            if key in PASS_TYPE_STATS:
                result[period_key][key] = {
                    "home": parse_pass_stat(home_val),
                    "away": parse_pass_stat(away_val),
                }
            elif key == "possession":
                home_int = int(re.sub(r"\D", "", str(home_val))) if home_val else 0
                away_int = int(re.sub(r"\D", "", str(away_val))) if away_val else 0
                result[period_key][key] = {"home": home_int, "away": away_int}
            else:
                def direct(val: Any) -> Any:
                    if val is None:
                        return None
                    if isinstance(val, (int, float)):
                        return val
                    s = str(val).strip()
                    if s == "-" or s == "":
                        return None
                    try:
                        return float(s)
                    except ValueError:
                        return s

                result[period_key][key] = {
                    "home": direct(home_val),
                    "away": direct(away_val),
                }

    return result


def _safe_optional(val, default=None):
    """Safe numeric conversion."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_stat(stats: Dict, key: str, side: str) -> Optional[float]:
    """Get a stat value safely from live match stats."""
    if not stats:
        return None
    period = stats.get("match", {})
    if not period:
        return None
    stat = period.get(key, {})
    if isinstance(stat, dict):
        return _safe_optional(stat.get(side))
    return None


def _infer_minute(summary: Optional[Dict], match_stats: Optional[Dict]) -> int:
    """Infer current match minute from summary events or stats."""
    if summary:
        events = summary.get("events", [])
        if events:
            # Use the last non-null minute
            for e in reversed(events):
                if e.get("minutes") is not None:
                    return e["minutes"]
    if match_stats:
        # Try to get minute from match stats
        periods = match_stats.get("match", {})
        for key in ("elapsed", "minute", "time"):
            if key in periods:
                return int(periods[key])
    return 0


# =============================================================================
# LIVE PROBABILITY ENGINE
# =============================================================================

def build_live_state(match_data: Dict, match_minute: int, injury_time: int = 0) -> Dict:
    """
    Build a clean, normalized live state snapshot from raw match data.
    This is the FOUNDATION — all probability calculations depend on it.
    """
    summary = match_data.get("summary") or {}
    events = summary.get("events", [])
    match_stats = match_data.get("match_stats") or {}
    # Normalize: stats may be at root or nested under ["match"]
    stats_root = match_stats.get("match", match_stats)
    lineups = match_data.get("lineups") or {}

    # --- Score: primary source match["scores"], fallback to events parsing ---
    match_scores = match_data.get("scores", {}) or {}
    _home_from_scores = match_scores.get("home")
    _away_from_scores = match_scores.get("away")

    # Card counters always from events (no card counts in scores)
    yellow_cards_home = 0
    yellow_cards_away = 0
    red_cards_home = 0
    red_cards_away = 0

    # If scores not in primary source, parse events
    if _home_from_scores is None or _away_from_scores is None:
        home_goals = 0
        away_goals = 0
        for e in events:
            e_type = e.get("type", "")
            e_text = str(e.get("text", "") or "")
            e_team = e.get("team", "")

            is_goal = (e_type == "goal") or ("goal" in e_text.lower() and "own" not in e_text.lower())
            if is_goal and e_team == "home":
                home_goals += 1
            elif is_goal and e_team == "away":
                away_goals += 1

            if e_type == "yellow_card" or "yellow" in e_text.lower():
                if e_team == "home":
                    yellow_cards_home += 1
                elif e_team == "away":
                    yellow_cards_away += 1

            if e_type == "red_card" or "red card" in e_text.lower():
                if e_team == "home":
                    red_cards_home += 1
                elif e_team == "away":
                    red_cards_away += 1

        # Also try commentary as semantic fallback if events gave no goals
        # Only use commentary if team names are available for reliable matching
        if home_goals == 0 and away_goals == 0:
            home_team_name = str(match_data.get("home_team_name") or "").lower()
            away_team_name = str(match_data.get("away_team_name") or "").lower()
            commentary_data = match_data.get("commentary") or {}
            commentary_items = commentary_data.get("commentary", []) or []
            for c in commentary_items:
                desc = str(c.get("description") or "").lower()
                minute = c.get("minutes")
                if "goal" in desc and "own" not in desc and minute is not None:
                    # Map to home/away using real team names — never "1st"/"2nd" as proxy
                    if home_team_name and home_team_name in desc:
                        home_goals += 1
                    elif away_team_name and away_team_name in desc:
                        away_goals += 1
    else:
        home_goals = _home_from_scores
        away_goals = _away_from_scores

    score = {"home": home_goals, "away": away_goals}
    goals_so_far = home_goals + away_goals

    # --- Stats from match_stats (safe access) ---
    def sstat(key, side):
        v = stats_root.get(key, {})
        return v.get(side) if isinstance(v, dict) else None

    xg_home = sstat("xg", "home") or 0.0
    xg_away = sstat("xg", "away") or 0.0
    xg_total = xg_home + xg_away

    shots_home = sstat("shots", "home") or 0
    shots_away = sstat("shots", "away") or 0
    shots_total = shots_home + shots_away

    sot_home = sstat("shots_on_target", "home") or 0
    sot_away = sstat("shots_on_target", "away") or 0

    bc_home = sstat("big_chances", "home") or 0
    bc_away = sstat("big_chances", "away") or 0

    corners_home = sstat("corners", "home") or 0
    corners_away = sstat("corners", "away") or 0

    yc_home = sstat("yellow_cards", "home") or yellow_cards_home or 0
    yc_away = sstat("yellow_cards", "away") or yellow_cards_away or 0
    rc_home = sstat("red_cards", "home") or red_cards_home or 0
    rc_away = sstat("red_cards", "away") or red_cards_away or 0

    poss_home = sstat("possession", "home") or 50.0
    poss_away = sstat("possession", "away") or 50.0

    # --- Lineup context ---
    home_lineup = lineups.get("home", {}) or {}
    away_lineup = lineups.get("away", {}) or {}

    missing_home = home_lineup.get("missing_players") or []
    missing_away = away_lineup.get("missing_players") or []

    # --- Derived features ---
    xg_diff = xg_home - xg_away
    time_remaining = max(90 - match_minute, 0)
    minute = match_minute

    # xG rates (goals per minute)
    xg_rate_home = xg_home / max(minute, 1)
    xg_rate_away = xg_away / max(minute, 1)

    # Score effect: winning team typically sits deeper
    # Using 1/(1 + 0.20*diff) — 3-0 → 0.625x for more aggressive sit-back
    score_diff = home_goals - away_goals
    score_effect_home = 1.0 / (1.0 + 0.20 * max(score_diff, 0))
    score_effect_away = 1.0 / (1.0 + 0.20 * max(-score_diff, 0))

    # Data quality check
    warnings = []
    if shots_total == 0 and minute > 10:
        warnings.append("no_shots_registered")
    if xg_total == 0 and shots_total > 5:
        warnings.append("xg_zero_with_shots_unusual")
    if minute > 45 and goals_so_far == 0 and xg_total < 0.5:
        warnings.append("very_low_action_late")

    # Data quality check — multi-signal, not just absence of warnings
    # high: full consistency across scores + stats + events + commentary + minute > 20
    # medium: base stats present but some inconsistency or missing signals
    # low: early minute, missing stats, or contradictory sources
    has_scores_primary = (_home_from_scores is not None and _away_from_scores is not None)
    has_base_stats = shots_total >= 3 and xg_total > 0
    has_rich_events = any((e.get("type") or e.get("description") or e.get("text")) for e in events)
    commentary_raw = match_data.get("commentary")
    commentary_items = commentary_raw.get("commentary", []) if isinstance(commentary_raw, dict) else (commentary_raw if isinstance(commentary_raw, list) else [])
    has_usable_commentary = any(c.get("minutes") is not None and c.get("description") for c in commentary_items)
    minute_ok = minute >= 10

    if has_scores_primary and has_base_stats and has_rich_events and has_usable_commentary and minute_ok:
        data_quality = "high"
    elif has_base_stats and minute_ok:
        data_quality = "medium"
    else:
        data_quality = "low"

    return {
        "minute": minute,
        "time_remaining": time_remaining,
        "score": score,
        "goals_so_far": goals_so_far,
        "xg": {"home": xg_home, "away": xg_away},
        "xg_total": xg_total,
        "xg_diff": xg_diff,
        "xg_rate_home": xg_rate_home,
        "xg_rate_away": xg_rate_away,
        "xgot": {"home": sstat("xgot", "home") or 0.0, "away": sstat("xgot", "away") or 0.0},
        "shots": {"home": shots_home, "away": shots_away},
        "shots_total": shots_total,
        "shots_on_target": {"home": sot_home, "away": sot_away},
        "big_chances": {"home": bc_home, "away": bc_away},
        "corners": {"home": corners_home, "away": corners_away},
        "yellow_cards": {"home": yc_home, "away": yc_away},
        "red_cards": {"home": rc_home, "away": rc_away},
        "possession": {"home": poss_home, "away": poss_away},
        "lineups": {
            "home": {"missing_count": len(missing_home), "red_card": bool(rc_home > 0), "substitutions_made": home_lineup.get("substitutions_made", 0)},
            "away": {"missing_count": len(missing_away), "red_card": bool(rc_away > 0), "substitutions_made": away_lineup.get("substitutions_made", 0)},
        },
        "score_effect_home": score_effect_home,
        "score_effect_away": score_effect_away,
        "data_quality": data_quality,
        "warnings": warnings,
        "events": events,
    }


# ---------------------------------------------------------------------------
# POISSON UTILITIES
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    """Probability of exactly k events given Poisson lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k > 100:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_at_least(k: int, lam: float) -> float:
    """P(X >= k) for Poisson via 1 - P(X < k) = 1 - sum(i=0 to k-1) PMF(i)."""
    if k <= 0:
        return 1.0
    # Compute CDF up to k-1 (P(X < k)) then subtract from 1
    total = 0.0
    for i in range(k):
        total += _poisson_pmf(i, lam)
        if total > 0.9999:
            break
    return min(1.0 - total, 1.0)


# ---------------------------------------------------------------------------
# POISSON GOAL ENGINE
# ---------------------------------------------------------------------------

BASELINE_RATE = 1.2 / 90  # ~0.013 goals per minute — floor for lambda


def poisson_goal_engine(state: Dict) -> Dict:
    """
    Unified goal probability engine.
    All goal-based markets (O/U, BTTS, next goal) derive from shared
    Poisson lambdas estimated from live state.
    """
    minute = state["minute"]
    time_remaining = state["time_remaining"]
    goals_so_far = state["goals_so_far"]
    score = state["score"]
    xg_home = state["xg"]["home"]
    xg_away = state["xg"]["away"]
    data_quality = state.get("data_quality", "medium")
    warnings = list(state.get("warnings", []))

    # --- Estimate per-team goal rates ---
    rate_home = max(xg_home / max(minute, 1), BASELINE_RATE)
    rate_away = max(xg_away / max(minute, 1), BASELINE_RATE)

    # Apply score effect (winning team typically sits deeper)
    rate_home *= state.get("score_effect_home", 1.0)
    rate_away *= state.get("score_effect_away", 1.0)

    # Enhance rates using advanced stats (shots on target, big chances)
    sot_home = state["shots_on_target"]["home"]
    sot_away = state["shots_on_target"]["away"]
    bc_home = state["big_chances"]["home"]
    bc_away = state["big_chances"]["away"]
    shots_home = state["shots"]["home"]
    shots_away = state["shots"]["away"]

    if shots_home > 0:
        rate_home *= (1.0 + 0.15 * (sot_home / shots_home))
    if shots_away > 0:
        rate_away *= (1.0 + 0.15 * (sot_away / shots_away))
    rate_home *= (1.0 + 0.10 * bc_home)
    rate_away *= (1.0 + 0.08 * min(bc_away, 3))

    # xGOT boost: quality of shots on target (better than raw xG alone)
    # xGOT = expected goals on target, more discriminative than raw xG
    xg_home = state["xg"]["home"]
    xg_away = state["xg"]["away"]
    xgot_home = state.get("xgot", {}).get("home", 0)
    xgot_away = state.get("xgot", {}).get("away", 0)
    if xg_home > 0:
        xgot_ratio_home = min(xgot_home / max(xg_home, 0.01), 3.0)  # cap at 3x
        rate_home *= (1.0 + 0.20 * (xgot_ratio_home - 1.0))
    if xg_away > 0:
        xgot_ratio_away = min(xgot_away / max(xg_away, 0.01), 3.0)
        rate_away *= (1.0 + 0.20 * (xgot_ratio_away - 1.0))

    # Time non-linearity: intensity decays slightly toward end
    time_factor = (time_remaining / 90) ** 0.85

    lambda_home = rate_home * time_remaining * time_factor
    lambda_away = rate_away * time_remaining * time_factor

    # Low data quality warning (don't reduce lambdas — keep expectations stable)
    if data_quality == "low":
        warnings.append("low_data_quality_adjustment")

    # --- Over/Under 2.5 ---
    goals_to_over = max(3 - goals_so_far, 0)
    if goals_to_over == 0:
        over_prob = 1.0
    else:
        lambda_total = lambda_home + lambda_away
        over_prob = _poisson_at_least(goals_to_over, lambda_total)

    under_prob = round(1.0 - over_prob, 4)

    # --- BTTS ---
    home_scored = score["home"] > 0
    away_scored = score["away"] > 0

    if home_scored and away_scored:
        btts_yes = 1.0
    elif home_scored:
        btts_yes = 1.0 - _poisson_pmf(0, lambda_away)
    elif away_scored:
        btts_yes = 1.0 - _poisson_pmf(0, lambda_home)
    else:
        p_home_scores = 1.0 - _poisson_pmf(0, lambda_home)
        p_away_scores = 1.0 - _poisson_pmf(0, lambda_away)
        btts_yes = p_home_scores * p_away_scores

    btts_no = round(1.0 - btts_yes, 4)

    # --- Next goal (competing risks with explicit no_more_goals) ---
    total_rate = lambda_home + lambda_away

    # Survival probability: no more goals scored
    next_no_more = math.exp(-total_rate)

    # Conditional on a goal being scored: home vs away
    if total_rate > 0 and next_no_more < 1.0:
        # P(home | goal) = lambda_home / (lambda_home + lambda_away)
        # P(away | goal) = lambda_away / (lambda_home + lambda_away)
        # P(goal) = 1 - next_no_more
        # So P(home) = (1 - next_no_more) * lambda_home / total_rate
        # And P(away) = (1 - next_no_more) * lambda_away / total_rate
        # This guarantees sum = 1.0
        next_home = (1.0 - next_no_more) * (lambda_home / total_rate)
        next_away = (1.0 - next_no_more) * (lambda_away / total_rate)
    else:
        next_home = 0.0
        next_away = 0.0
        next_no_more = 1.0

    return {
        "over_under_2_5": {
            "probabilities": {"over": round(over_prob, 4), "under": under_prob},
            "model_version": "poisson_v1",
            "calibrated": False,
            "inputs_quality": data_quality,
            "top_factors": ["xg_remaining", "goals_so_far", "minute"],
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
            "warnings": warnings,
        },
        "btts": {
            "probabilities": {"yes": round(btts_yes, 4), "no": btts_no},
            "model_version": "poisson_v1",
            "calibrated": False,
            "inputs_quality": data_quality,
            "top_factors": ["score", "lambda_home", "lambda_away"],
            "warnings": warnings,
        },
        "next_goal": {
            "probabilities": {
                "home": round(next_home, 4),
                "away": round(next_away, 4),
                "no_more_goals": round(next_no_more, 4),
            },
            "model_version": "poisson_v1",
            "calibrated": False,
            "inputs_quality": data_quality,
            "top_factors": ["lambda_home", "lambda_away", "total_rate", "minute"],
            "warnings": warnings,
        },
        "_internal": {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
        }
    }


# ---------------------------------------------------------------------------
# PREMATCH + LIVE 1X2
# ---------------------------------------------------------------------------

def compute_prematch_priors(odds_data: Dict, home_team_results: Dict, away_team_results: Dict) -> Dict:
    """
    Compute pre-match 1X2 priors from closing odds (preferred) or Elo fallback.
    Returns: {"1x2": {"home": float, "draw": float, "away": float, "source": str}}
    """
    odds_home = odds_data.get("odds_home")
    odds_draw = odds_data.get("odds_draw")
    odds_away = odds_data.get("odds_away")

    if odds_home and odds_draw and odds_away:
        raw = [1/odds_home, 1/odds_draw, 1/odds_away]
        total = sum(raw)
        if total > 0:
            return {
                "1x2": {
                    "home": round(raw[0] / total, 4),
                    "draw": round(raw[1] / total, 4),
                    "away": round(raw[2] / total, 4),
                    "source": "closing_odds",
                }
            }

    def elo_from_results(results: Dict) -> float:
        matches = results.get("matches", [])
        if not matches:
            return 0.5
        pts = results.get("basic_stats", {}).get("points", 0)
        n = len(matches)
        return min(pts / (n * 3), 1.0)

    elo_home = elo_from_results(home_team_results)
    elo_away = elo_from_results(away_team_results)

    elo_home_adj = elo_home + 0.06
    elo_away_adj = elo_away

    elo_diff = elo_home_adj - elo_away_adj
    p_home_win = 1.0 / (1.0 + 10 ** (-elo_diff * 4))
    draw_factor = max(0.15, min(0.35, 0.25 - abs(elo_diff) * 0.5))
    p_draw = draw_factor
    remaining = 1.0 - p_draw
    p_home = p_home_win * remaining
    p_away = remaining - p_home

    return {
        "1x2": {
            "home": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away": round(p_away, 4),
            "source": "elo_fallback",
        }
    }


def _squash(x: float) -> float:
    """Logistic squash: maps any real number to [0, 1] centered at 0.5."""
    return 1.0 / (1.0 + math.exp(-x))


def update_1x2_prior(prematch_prior: Dict, live_state: Dict) -> Dict:
    """
    Update pre-match 1X2 prior with live evidence using logistic adjustments.
    """
    prior = prematch_prior.get("1x2", {"home": 0.33, "draw": 0.33, "away": 0.33})

    minute = live_state["minute"]
    score_diff = live_state["score"]["home"] - live_state["score"]["away"]
    xg_diff = live_state["xg_diff"]
    time_factor = min(minute / 90, 1.0)

    # Squash-based adjustments: no linear explosion, soft saturation
    # Combined signal: xG diff (per-unit) + score diff (weighted by time)
    signal = xg_diff * 1.5 + score_diff * 2.0 * time_factor
    adjustment = _squash(signal) - 0.5  # shift from [0,1] to [-0.5, +0.5]

    home = prior["home"] + adjustment
    away = prior["away"] - adjustment

    home = max(home, 0.05)
    away = max(away, 0.05)

    total = home + prior["draw"] + away
    if total > 0:
        home = round(home / total, 4)
        draw = round(prior["draw"] / total, 4)
        away = round(away / total, 4)
    else:
        home, draw, away = 0.33, 0.33, 0.34

    confidence = "low"
    if minute >= 45:
        confidence = "high"
    elif minute >= 30:
        confidence = "medium"

    return {
        "probabilities": {"home": home, "draw": draw, "away": away},
        "model_version": "prior_update_v1",
        "calibrated": False,
        "inputs_quality": live_state.get("data_quality", "medium"),
        "top_factors": ["prematch_prior", "xg_diff", "score_diff", "minute"],
        "prematch_source": prematch_prior.get("1x2", {}).get("source", "unknown"),
        "confidence": confidence,
        "warnings": live_state.get("warnings", []),
    }


def compute_live_analysis(match_data: Dict, prematch_prior: Dict = None) -> Optional[Dict]:
    """
    Compute live probability signals for in-progress matches.
    Uses build_live_state → poisson_goal_engine + update_1x2_prior.
    """
    if not match_data:
        return None

    if prematch_prior is None:
        prematch_prior = {"1x2": {"home": 0.333, "draw": 0.334, "away": 0.333, "source": "default"}}

    summary = match_data.get("summary")
    match_stats = match_data.get("match_stats") or {}

    match_minute = _infer_minute(summary, match_stats)
    injury_time = 0
    if summary:
        inj = summary.get("injuryTime")
        if inj is not None:
            injury_time = inj

    state = build_live_state(match_data, match_minute, injury_time)

    goal_markets = poisson_goal_engine(state)

    x2f = update_1x2_prior(prematch_prior, state)

    yc_home = state["yellow_cards"]["home"]
    yc_away = state["yellow_cards"]["away"]
    cards_total = yc_home + yc_away
    # Require more cards to trigger signals — avoid false positives on sparse card data
    cards_per_min = cards_total / max(state["minute"], 1)
    cards_lean = "over" if cards_per_min > 0.06 else "under" if cards_per_min < 0.015 else None
    cards_signal = "aggressive_match" if cards_per_min > 0.06 else "disciplined" if cards_per_min < 0.015 else "baseline"

    corners_diff = state["corners"]["home"] - state["corners"]["away"]
    shots_diff = state["shots"]["home"] - state["shots"]["away"]
    if corners_diff > 2 or shots_diff > 4:
        corner_lean = "home"
        corner_signal = "home_pressure"
    elif corners_diff < -2 or shots_diff < -4:
        corner_lean = "away"
        corner_signal = "away_pressure"
    else:
        corner_lean = None
        corner_signal = "balanced"

    x2f_probs = x2f["probabilities"]
    return {
        "total_goals_over_25": goal_markets["over_under_2_5"],
        "total_goals_under_25": {
            "probabilities": {"under": goal_markets["over_under_2_5"]["probabilities"]["under"]},
            "model_version": goal_markets["over_under_2_5"]["model_version"],
            "calibrated": False,
            "inputs_quality": state["data_quality"],
            "warnings": goal_markets["over_under_2_5"]["warnings"],
        },
        "btts_yes": {
            "probabilities": {"yes": goal_markets["btts"]["probabilities"]["yes"]},
            "model_version": goal_markets["btts"]["model_version"],
            "calibrated": False,
            "inputs_quality": state["data_quality"],
            "top_factors": goal_markets["btts"]["top_factors"],
            "warnings": goal_markets["btts"]["warnings"],
        },
        "btts_no": {
            "probabilities": {"no": goal_markets["btts"]["probabilities"]["no"]},
            "model_version": goal_markets["btts"]["model_version"],
            "calibrated": False,
            "inputs_quality": state["data_quality"],
            "warnings": goal_markets["btts"]["warnings"],
        },
        "next_goal_home": {
            "probabilities": {"home": goal_markets["next_goal"]["probabilities"]["home"]},
            "model_version": goal_markets["next_goal"]["model_version"],
            "calibrated": False,
            "inputs_quality": state["data_quality"],
            "top_factors": goal_markets["next_goal"]["top_factors"],
            "warnings": goal_markets["next_goal"]["warnings"],
        },
        "next_goal_away": {
            "probabilities": {"away": goal_markets["next_goal"]["probabilities"]["away"]},
            "model_version": goal_markets["next_goal"]["model_version"],
            "calibrated": False,
            "inputs_quality": state["data_quality"],
            "top_factors": goal_markets["next_goal"]["top_factors"],
            "warnings": goal_markets["next_goal"]["warnings"],
        },
        "next_goal_no_more": {
            "probabilities": {"no_more_goals": goal_markets["next_goal"]["probabilities"]["no_more_goals"]},
            "model_version": goal_markets["next_goal"]["model_version"],
            "calibrated": False,
            "inputs_quality": state["data_quality"],
            "warnings": goal_markets["next_goal"]["warnings"],
        },
        "next_corner": {
            "lean": corner_lean,
            "signal": corner_signal,
            "confidence": "low",
            "mode": "signal_only",
        },
        "total_cards": {
            "lean": cards_lean,
            "signal": cards_signal,
            "confidence": "low",
            "mode": "signal_only",
        },
        "1x2_final": x2f,
        "match_minute": state["minute"],
        "current_score": state["score"],
        "warnings": state["warnings"],
    }


def normalize_match_status(status_data: Optional[Dict]) -> str:
    """
    Normalize match status object to a flat string:
    'notstarted', 'inprogress', 'finished'.
    """
    if not status_data:
        return "notstarted"

    is_started = status_data.get("is_started", False)
    is_finished = status_data.get("is_finished", False)
    is_in_progress = status_data.get("is_in_progress", False)

    if is_finished:
        return "finished"
    elif is_in_progress:
        return "inprogress"
    elif is_started:
        return "inprogress"  # started but not yet in-progress = kickoff moment
    else:
        return "notstarted"


def parse_pass_stat(value: Any) -> Dict:
    """
    Parse pass-type stat strings like '81% (256/316)'.
    Returns {pct: float, completed: int, attempted: int}.
    """
    if value is None:
        return {"pct": None, "completed": None, "attempted": None}
    if isinstance(value, (int, float)):
        return {"pct": float(value), "completed": None, "attempted": None}
    s = str(value)
    pct_match = re.match(r"(\d+(?:\.\d+)?)%", s)
    pct = float(pct_match.group(1)) if pct_match else None
    bracket_match = re.search(r"\((\d+)/(\d+)\)", s)
    if bracket_match:
        completed = int(bracket_match.group(1))
        attempted = int(bracket_match.group(2))
    else:
        completed = attempted = None
    return {"pct": pct, "completed": completed, "attempted": attempted}


def compute_team_player_aggregates(matches: List[Dict]) -> Dict:
    """
    Aggregate player stats across all historical matches for one team.
    Collects all player records from each match's player_stats,
    orienting by team_is_home to select the right player list per match,
    then computes totals across the full sample.
    """
    all_home_players: List[Dict] = []
    all_away_players: List[Dict] = []

    for match in matches:
        ps = match.get("player_stats", {})
        if not ps or ps.get("warnings"):
            continue
        is_home = match.get("team_is_home", True)
        if is_home:
            # Analyzed team was home in this match → their players are in home_players
            all_home_players.extend(ps.get("home_players", []))
            # Opponent (away) players
            all_away_players.extend(ps.get("away_players", []))
        else:
            # Analyzed team was away in this match.
            # normalize_player_stats uses api_home_id=analyzed_team, api_away_id=opponent,
            # so analyzed team's players are in away_players, opponent's in home_players.
            all_away_players.extend(ps.get("away_players", []))
            all_home_players.extend(ps.get("home_players", []))

    return {
        "as_historical_home": compute_player_aggregates(all_home_players, include_individual=True),
        "as_historical_away": compute_player_aggregates(all_away_players, include_individual=True),
    }


def compute_player_aggregates(players: List[Dict], include_individual: bool = False) -> Dict:
    """
    Aggregate stats across all players of a team within a single match.
    Produces totals, per-90-minute rates, and distribution breakdowns.
    If include_individual=True, also aggregates per-stat-key across all players.
    """
    if not players:
        return {}

    def g(player: Dict, key: str) -> float:
        v = player.get(key, 0)
        return float(v) if v is not None else 0.0

    total_minutes = sum(g(p, "minutes") for p in players)
    n = len(players)

    totals = {
        "minutes_total": total_minutes,
        "goals_total": sum(g(p, "goals") for p in players),
        "assists_total": sum(g(p, "assists") for p in players),
        "shots_total": sum(g(p, "shots") for p in players),
        "shots_on_target_total": sum(g(p, "shots_on_target") for p in players),
        "key_passes_total": sum(g(p, "key_passes") for p in players),
        "tackles_won_total": sum(g(p, "tackles_won") for p in players),
        "interceptions_total": sum(g(p, "interceptions") for p in players),
        "ball_recoveries_total": sum(g(p, "ball_recoveries") for p in players),
        "yellow_cards_total": sum(g(p, "yellow_cards") for p in players),
        "red_cards_total": sum(g(p, "red_cards") for p in players),
    }

    total_90 = total_minutes / PARTY_MINUTES
    totals["goals_per_90"] = round(totals["goals_total"] / total_90, 2) if total_90 > 0 else None
    totals["assists_per_90"] = round(totals["assists_total"] / total_90, 2) if total_90 > 0 else None
    totals["shots_per_90"] = round(totals["shots_total"] / total_90, 2) if total_90 > 0 else None

    pos_count: Dict[str, int] = {}
    for p in players:
        pos = p.get("position") or "Unknown"
        pos_count[pos] = pos_count.get(pos, 0) + 1
    totals["position_distribution"] = pos_count

    lineup_count = sum(1 for p in players if p.get("in_base_lineup"))
    totals["in_base_lineup_count"] = lineup_count
    totals["substitute_count"] = n - lineup_count

    result = {"general": totals}

    if include_individual:
        # Stat key → category mapping
        STAT_CATEGORIES: Dict[str, List[str]] = {
            "offense":    ["GOALS", "EXPECTED_GOALS", "ASSISTS_GOAL", "EXPECTED_ASSISTS",
                           "SHOTS_TOTAL", "SHOTS_ON_TARGET", "BIG_CHANCES_CREATED", "BIG_CHANCES_MISSED"],
            "creation":   ["KEY_PASSES", "FINAL_THIRD_ENTRIES_TOTAL", "BOX_ENTRIES", "THROUGH_BALLS"],
            "possession": ["TOUCHES_TOTAL", "MATCH_MINUTES_PLAYED", "PASSES_TOTAL"],
            "defense":    ["DUELS_WON", "DUELS_TOTAL", "DUELS_EFFICIENCY", "TACKLES_WON",
                           "INTERCEPTIONS", "BALL_RECOVERIES"],
            "efficiency": ["PASSES_ACCURACY", "LONG_BALLS_ACCURACY", "CROSSES_ACCURACY",
                          "DRIBBLES_EFFICIENCY"],
            "discipline": ["FOULS_COMMITTED", "FOULS_SUFFERED", "CARDS_YELLOW", "CARDS_RED",
                           "TURNOVERS", "ERRORS_LEAD_TO_SHOT", "ERRORS_LEAD_TO_GOAL"],
            "goalkeeping":["SAVES_TOTAL", "GOALS_CONCEDED", "GOALS_PREVENTED",
                           "EXPECTED_GOALS_ON_TARGET_FACED", "BIG_CHANCES_SAVED"],
        }
        # Flat whitelist of all allowed keys
        ALLOWED_KEYS: set = {k for cat in STAT_CATEGORIES.values() for k in cat}

        player_stats_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
        player_positions: Dict[str, str] = {}
        for p in players:
            player_name = p.get("name") or p.get("player_name") or f"player_{id(p)}"
            player_positions[player_name] = p.get("position") or "Unknown"
            if player_name not in player_stats_map:
                player_stats_map[player_name] = {}
            player_stats = p.get("stats", {})
            if not isinstance(player_stats, dict):
                continue
            for stat_key, stat_data in player_stats.items():
                if stat_key not in ALLOWED_KEYS:
                    continue
                if not isinstance(stat_data, dict):
                    continue
                raw_val = stat_data.get("raw_value") if stat_data.get("raw_value") is not None else stat_data.get("value")
                if raw_val is None:
                    continue
                try:
                    numeric_val = float(raw_val)
                except (ValueError, TypeError):
                    continue
                if stat_key not in player_stats_map[player_name]:
                    player_stats_map[player_name][stat_key] = {"sum": 0.0, "count": 0}
                player_stats_map[player_name][stat_key]["sum"] += numeric_val
                player_stats_map[player_name][stat_key]["count"] += 1

        individual = {}
        for player_name, stat_aggregates in player_stats_map.items():
            player_agg: Dict[str, Dict[str, Any]] = {}
            player_agg["position"] = player_positions.get(player_name, "Unknown")
            for cat_key, cat_keys in STAT_CATEGORIES.items():
                cat_data: Dict[str, Any] = {}
                for stat_key in cat_keys:
                    agg = stat_aggregates.get(stat_key)
                    if not agg:
                        continue
                    per_90 = round(agg["sum"] / total_90, 2) if total_90 > 0 else None
                    cat_data[stat_key] = {
                        "sum": round(agg["sum"], 2),
                        "per_90": per_90,
                        "matches": agg["count"],
                    }
                if cat_data:
                    player_agg[cat_key] = cat_data

            # Derived stats per player
            derived: Dict[str, Any] = {}
            goals     = stat_aggregates.get("GOALS", {}).get("sum", 0)
            xg        = stat_aggregates.get("EXPECTED_GOALS", {}).get("sum", 0)
            assists   = stat_aggregates.get("ASSISTS_GOAL", {}).get("sum", 0)
            xa        = stat_aggregates.get("EXPECTED_ASSISTS", {}).get("sum", 0)
            shots     = stat_aggregates.get("SHOTS_TOTAL", {}).get("sum", 0)
            duels_won = stat_aggregates.get("DUELS_WON", {}).get("sum", 0)
            duels_tot = stat_aggregates.get("DUELS_TOTAL", {}).get("sum", 0)
            passes_tot= stat_aggregates.get("PASSES_TOTAL", {}).get("sum", 0)

            if shots > 0:
                derived["xg_per_shot"] = round(xg / shots, 3)
            if xg > 0:
                derived["goals_minus_xg"] = round(goals - xg, 2)
            if xa > 0:
                derived["xa_minus_assists"] = round(assists - xa, 2)
            if duels_tot > 0:
                derived["duels_win_pct"] = round(duels_won / duels_tot * 100, 1)
            if total_90 > 0:
                actions_per_90 = sum(
                    stat_aggregates.get(k, {}).get("sum", 0)
                    for k in ["TOUCHES_TOTAL", "PASSES_TOTAL", "SHOTS_TOTAL", "KEY_PASSES"]
                )
                derived["actions_per_90"] = round(actions_per_90 / total_90, 2)

            if derived:
                player_agg["derived"] = derived

            if player_agg:
                individual[player_name] = player_agg

        result["individual"] = individual

    return result



def normalize_player_stats(player_data: Any, home_team_id: str, away_team_id: str) -> Dict:
    """
    Normalize per-player stats for the current match.
    Extracts ALL available stat keys from the raw API response (not just a fixed set).
    Each player's `stats` dict contains every stat key the API provides for that match.
    No aggregations — for the current match only raw normalized data per player.
    """
    if not player_data:
        return {"home_players": [], "away_players": [], "warnings": ["Player stats not available [N/A]"]}

    all_players = player_data.get("players", []) if isinstance(player_data, dict) else []

    home_players = []
    away_players = []

    for p in all_players:
        pid = p.get("player_id")
        tid = p.get("team_id")
        raw_stats = p.get("stats", {})

        # Extract every stat key present in the raw dict (dynamic — no fixed list)
        stats_record: Dict[str, Any] = {}
        if isinstance(raw_stats, dict):
            for stat_key, stat_val in raw_stats.items():
                if not isinstance(stat_val, dict):
                    continue
                raw_v = stat_val.get("raw_value") or stat_val.get("value")
                str_v = str(raw_v) if raw_v is not None else None
                rank = stat_val.get("rank")

                if str_v is not None and re.search(r"\d+%?\s*\(", str_v):
                    parsed = parse_pass_stat(str_v)
                    stats_record[stat_key] = {
                        "value": parsed.get("pct"),
                        "pct": parsed.get("pct"),
                        "completed": parsed.get("completed"),
                        "attempted": parsed.get("attempted"),
                        "rank": rank,
                    }
                else:
                    if str_v is not None:
                        try:
                            num_val = int(str_v)
                        except (ValueError, TypeError):
                            try:
                                num_val = float(str_v)
                            except (ValueError, TypeError):
                                num_val = str_v
                    else:
                        num_val = None
                    stats_record[stat_key] = {
                        "value": num_val,
                        "rank": rank,
                    }

        # Fixed extraction — always present for backwards compatibility
        def g(key):
            if not isinstance(raw_stats, dict):
                return 0
            s = raw_stats.get(key, {})
            v = s.get("value", "0") if isinstance(s, dict) else "0"
            try:
                return int(v)
            except (ValueError, TypeError):
                return 0

        player_rec = {
            "player_id": pid,
            "name": p.get("name"),
            "short_name": p.get("short_name"),
            "position": p.get("position"),
            "in_base_lineup": p.get("in_base_lineup"),
            "is_goalkeeper": p.get("is_goalkeeper", False),
            "stats": stats_record,
            "goals": g("GOALS"),
            "assists": g("ASSISTS_GOAL"),
            "shots": g("TOTAL_SHOTS"),
            "shots_on_target": g("SHOTS_ON_TARGET_STATE"),
            "key_passes": g("KEY_PASSES"),
            "tackles_won": g("TACKLES_WON"),
            "interceptions": g("INTERCEPTIONS"),
            "ball_recoveries": g("BALL_RECOVERIES"),
            "yellow_cards": g("CARDS_YELLOW"),
            "red_cards": g("CARDS_RED"),
            "minutes": g("MINUTES"),
        }

        if tid == home_team_id:
            home_players.append(player_rec)
        elif tid == away_team_id:
            away_players.append(player_rec)

    return {
        "home_players": home_players,
        "away_players": away_players,
        "warnings": [],
    }


def normalize_lineups(lineup_data: Any) -> Dict:
    """
    Normalize lineups + extract missingPlayers for each team.
    """
    if not lineup_data:
        return {"home": None, "away": None, "warnings": ["Lineup data not available [N/A]"]}

    result = {"home": None, "away": None, "warnings": []}
    
    def parse_lineups(lineups: List, include_reason: bool = False) -> List[Dict]:
        parsed = []
        for m in lineups:
            player_data = {
                "country": m.get("country_name"),
                "player_id": m.get("player_id"),
                "name": m.get("name"),
            }
            if include_reason:
                player_data["reason"] = m.get("reason")
            parsed.append(player_data)
        return parsed

    for team_block in lineup_data:
        unsureMissingPlayers = team_block.get("unsureMissingPlayers", [])
        predictedLineups = team_block.get("predictedLineups", [])
        startingLineups = team_block.get("startingLineups", [])
        missingPlayers = team_block.get("missingPlayers", [])
        formation = team_block.get("predictedFormation")
        substitutes = team_block.get("substitutes", [])
        side = team_block.get("side")

        team_data = {
            "missing_players": parse_lineups(missingPlayers, True),
            "predicted_lineups": [],
            "formation": formation,
            "starting_lineups": [],
            "unsure_missing": [],
            "substitutes": [],
        }
        
        if startingLineups:
            team_data["starting_lineups"] = parse_lineups(startingLineups)
            team_data["substitutes"] = parse_lineups(substitutes)
        elif predictedLineups:
            team_data["unsure_missing"] = parse_lineups(unsureMissingPlayers, True)
            team_data["predicted_lineups"] = parse_lineups(predictedLineups)

        if side == "home":
            result["home"] = team_data
        elif side == "away":
            result["away"] = team_data

    return result

def normalize_standings(standings_data: List, team_ids: set) -> Dict:
    """
    Extract standings rows for the two teams in the match.
    """
    if not standings_data:
        return {"teams": {}, "warnings": ["Standings not available [N/A]"]}

    teams = {}
    for row in standings_data:
        tid = row.get("team_id")
        if tid in team_ids:
            teams[tid] = {
                "position": None,  # will be set by enumerate
                "name": row.get("name"),
                "matches_played": row.get("matches_played"),
                "wins": row.get("wins"),
                "draws": row.get("draws"),
                "losses": row.get("losses"),
                "goals": row.get("goals"),
                "goal_difference": row.get("goal_difference"),
                "points": row.get("points"),
            }

    # Assign positions
    sorted_teams = sorted(standings_data, key=lambda x: x.get("points", 0), reverse=True)
    for idx, row in enumerate(sorted_teams):
        tid = row.get("team_id")
        if tid in teams and teams[tid]:
            teams[tid]["position"] = idx + 1

    return {"teams": teams, "warnings": []}


def normalize_overunder_st(ou_data: List, team_ids: set) -> Dict:
    """Extract Over/Under standings for both teams."""
    if not ou_data:
        return {"teams": {}, "warnings": ["Over/Under standings not available [N/A]"]}

    teams = {}
    for row in ou_data:
        tid = row.get("team_id")
        if tid in team_ids:
            teams[tid] = {
                "name": row.get("name"),
                "matches_played": row.get("matches_played"),
                "over": row.get("over"),
                "under": row.get("under"),
                "average_goals": row.get("average_goals_per_match"),
            }
    return {"teams": teams, "warnings": []}


def normalize_form_st(form_data: List, team_ids: set) -> Dict:
    """Extract form standings (last 5 matches) for both teams."""
    if not form_data:
        return {"teams": {}, "warnings": ["Form standings not available [N/A]"]}

    teams = {}
    for row in form_data:
        tid = row.get("team_id")
        if tid in team_ids:
            teams[tid] = {
                "name": row.get("name"),
                "matches_played": row.get("matches_played"),
                "wins": row.get("wins"),
                "draws": row.get("draws"),
                "losses": row.get("losses"),
                "goals": row.get("goals"),
                "goal_difference": row.get("goal_difference"),
                "points": row.get("points"),
            }
    return {"teams": teams, "warnings": []}


def normalize_top_scorers(scorers_data: List, team_ids: set) -> Dict:
    """Extract top scorers for the tournament, filtered by the two teams."""
    if not scorers_data:
        return {"home_scorers": [], "away_scorers": [], "warnings": ["Top scorers not available [N/A]"]}

    home_scorers = []
    away_scorers = []

    for s in scorers_data:
        tid = s.get("team_id")
        rec = {
            "name": s.get("player_name"),
            "player_id": s.get("player_id"),
            "team": s.get("team_name"),
            "goals": s.get("goals"),
            "assists": s.get("assists"),
        }
        if tid == team_ids.get("home"):
            home_scorers.append(rec)
        elif tid == team_ids.get("away"):
            away_scorers.append(rec)

    return {
        "home_scorers": sorted(home_scorers, key=lambda x: x.get("goals", 0), reverse=True),
        "away_scorers": sorted(away_scorers, key=lambda x: x.get("goals", 0), reverse=True),
        "warnings": [],
    }


# =============================================================================
# AGGREGATE / CROSS-FUNCTION INDICATORS
# =============================================================================

def detect_h2h_inconsistencies(h2h: Dict) -> List[str]:
    """Check H2H for suspicious patterns."""
    warnings = []
    matches = h2h.get("matches", [])
    if not matches:
        return warnings

    # Check for future dates (shouldn't happen but check anyway)
    now = datetime.now().timestamp()
    future = [r for r in matches if r.get("timestamp", 0) > now]
    if future:
        warnings.append(f"H2H contains {len(future)} future-dated matches (possible data issue)")

    # Check for very high-scoring matches (potential data error if avg > 5)
    total = h2h.get("total_matches", 0)
    total_goals_sum = h2h.get("total_goals", 0)
    if total > 0 and total_goals_sum / total > MAX_AVG_GOALS_H2H:
        warnings.append(f"H2H avg goals unusually high: {round(total_goals_sum / total, 2)}")

    return warnings


def validate_data_completeness(ctx: Dict) -> Dict[str, List[str]]:
    """
    Cross-check: do stats suggest one team is stronger but odds don't reflect?
    Returns a dict with warnings categorized by section.
    """
    result: Dict[str, List[str]] = {
        "home_team_results": [],
        "away_team_results": [],
        "odds": [],
    }

    odds = ctx.get("odds", {})
    team_h = ctx.get("home_team_results", {}).get("basic_stats", {})
    team_a = ctx.get("away_team_results", {}).get("basic_stats", {})

    prob_h = odds.get("prob_home")
    if prob_h and team_h.get("home_gf_avg") and team_a.get("home_gf_avg"):
        form_diff = team_h["home_gf_avg"] - team_a["home_gf_avg"]
        if form_diff > FORM_DIFF_THRESHOLD and prob_h < 40:
            result["odds"].append("Market undervaluing home team despite stronger recent form [IND]")
        elif form_diff < -FORM_DIFF_THRESHOLD and prob_h > 60:
            result["odds"].append("Market overvaluing home team despite weaker recent form [IND]")

    return result


def _compute_top_level_stats(matches: List[Dict]) -> Dict:
    """
    Compute top-level summary stats (total_matches, wins, draws, losses,
    goals_for, goals_against, total_goals, both_teams_scored) from a list
    of matches. Supports two formats:
    - home_team_results/away_results: goals_for, goals_against, total_goals
    - h2h: home_score, away_score, team_is_home (computes goals_for from orientation)
    """
    wins = draws = losses = 0
    goals_for = goals_against = total_goals = 0
    btts_count = 0
    n = len(matches)

    for m in matches:
        # h2h format: home_score, away_score, team_is_home
        if "home_score" in m and "away_score" in m:
            hs = m.get("home_score", 0) or 0
            aw = m.get("away_score", 0) or 0
            is_home = m.get("team_is_home", True)
            gf = hs if is_home else aw
            ga = aw if is_home else hs
            tg = hs + aw
        # team results format: goals_for, goals_against
        else:
            gf = m.get("goals_for", 0) or 0
            ga = m.get("goals_against", 0) or 0
            tg = m.get("total_goals", gf + ga)

        goals_for += gf
        goals_against += ga
        total_goals += tg
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1
        if gf > 0 and ga > 0:
            btts_count += 1

    return {
        "total_matches": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "total_goals": total_goals,
        "both_teams_scored": f"{btts_count}/{n}" if n > 0 else None,
    }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def build_context(event_id: str, home_team_id: str, away_team_id: str) -> Dict:
    """
    Main function. Fetches all endpoints, normalizes, aggregates, returns final_context.
    """
    final = {
        "meta": {
            "event_id": event_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "generated_at": datetime.now().isoformat(),
        },
        "match": {"warnings": []},
        "odds": {},
        "implied_probs": {},
        "h2h": {},
        "home_team_results": {},
        "away_team_results": {},
        "standings": {},
        "overunder_standings": {},
        "form_standings": {},
        "top_scorers": {},
        "tournament_top_scorers": {},
        "indicators": {},
        "prematch_priors": {},
    }

    # -------------------------------------------------------------------------
    # 1. MATCH DETAILS (blocking)
    # -------------------------------------------------------------------------
    details = fetch_match_details(event_id)

    if not details:
        final["match"]["warnings"].append("CRITICAL: Could not fetch match details. Analysis not viable.")
        return final

    home_epid = details.get("home_team", {}).get("event_participant_id")
    away_epid = details.get("away_team", {}).get("event_participant_id")
    home_url = details.get("home_team", {}).get("team_url")
    away_url = details.get("away_team", {}).get("team_url")
    home_name = details.get("home_team", {}).get("name")
    away_name = details.get("away_team", {}).get("name")
    tournament_id = details.get("tournament", {}).get("tournament_id")
    tournament_stage_id = details.get("tournament", {}).get("tournament_stage_id")
    tournament_name = details.get("tournament", {}).get("name")
    match_timestamp = details.get("timestamp")
    country = details.get("country", {}).get("name")
    referee = details.get("referee")

    final["match"] = {
        "event_id": event_id,
        "home_team": {"id": home_team_id, "name": home_name, "event_participant_id": home_epid},
        "away_team": {"id": away_team_id, "name": away_name, "event_participant_id": away_epid},
        "tournament": {"id": tournament_id, "stage_id": tournament_stage_id, "name": tournament_name},
        "country": country,
        "referee": referee,
        "timestamp": match_timestamp,
        "datetime": datetime.fromtimestamp(match_timestamp).isoformat() if match_timestamp else None,
        "status": normalize_match_status(details.get("match_status")),
        "scores": details.get("scores"),
        "warnings": [],
    }

    team_ids = {"home": home_team_id, "away": away_team_id}

    # -------------------------------------------------------------------------
    # 2. ODDS
    # -------------------------------------------------------------------------
    odds_raw = fetch_match_odds(event_id)
    # print(f"Fetched raw odds data: {odds_raw}")
    if odds_raw:
        final["odds"] = normalize_odds(odds_raw, home_epid, away_epid)
        final["implied_probs"] = normalize_implied_probs(final["odds"])
    else:
        final["odds"]["warnings"] = ["Odds not available [N/A]"]

    # -------------------------------------------------------------------------
    # 3. TEAM RESULTS (historical form) — H2H is built from these below
    # -------------------------------------------------------------------------
    final["home_team_results"] = {"warnings": [f"Team results for {home_name} not available [N/A]"]}
    tr_home = fetch_team_results(home_team_id)
    if tr_home is not None:
        final["home_team_results"] = normalize_team_results(tr_home, home_name, home_team_id)

    final["away_team_results"] = {"warnings": [f"Team results for {away_name} not available [N/A]"]}
    tr_away = fetch_team_results(away_team_id)
    if tr_away is not None:
        final["away_team_results"] = normalize_team_results(tr_away, away_name, away_team_id)

    # -------------------------------------------------------------------------
    # PREMATCH PRIORS (for live 1X2) — uses odds + team results
    # -------------------------------------------------------------------------
    final["prematch_priors"] = compute_prematch_priors(
        final["odds"],
        final["home_team_results"],
        final["away_team_results"],
    )

    # -------------------------------------------------------------------------
    # 4. H2H — built from team_results (no extra API call)
    # -------------------------------------------------------------------------
    h2h = build_h2h_from_results(
        final["home_team_results"],
        final["away_team_results"],
        home_name,
        away_name,
    )
    final["h2h"] = h2h

    # --- ADVANCED STATS: Fetch in parallel for all historical matches ---
    # Deduplicate AND filter to past matches only (skip future fixtures)
    now = datetime.now().timestamp()
    seen = set()
    past_match_ids = []
    for match_list in [
        final["home_team_results"].get("matches", []),
        final["away_team_results"].get("matches", []),
    ]:
        for m in match_list:
            ts = m.get("timestamp", 0)
            mid = m.get("match_id")
            if mid and ts and ts < now and mid not in seen:
                seen.add(mid)
                past_match_ids.append(mid)

    all_match_ids = past_match_ids

    stats_map: Dict[str, Any] = {}
    player_stats_map: Dict[str, Any] = {}

    def _fetch_with_jitter(mid: str, fetch_func: Callable[..., Any]) -> tuple:
        time.sleep(API_SLEEP_BETWEEN + random.uniform(0, 0.2))
        try:
            result = fetch_func(mid)
        except Exception as e:
            print(f"[WARNING] {fetch_func.__name__}({mid}) failed: {e}", file=sys.stderr)
            result = None
        return mid, result

    if all_match_ids:
        time.sleep(API_SLEEP_INITIAL)

        # Fetch stats in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_with_jitter, mid, fetch_match_stats): mid for mid in all_match_ids}
            for future in as_completed(futures):
                mid, result = future.result()
                stats_map[mid] = result

        # Fetch player stats in parallel
        time.sleep(API_SLEEP_INITIAL)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_with_jitter, mid, fetch_match_player_stats): mid for mid in all_match_ids}
            for future in as_completed(futures):
                mid, result = future.result()
                player_stats_map[mid] = result

    def _enrich_match(
        match: Dict, stats_map: Dict, player_stats_map: Dict,
        home_tid: str, away_tid: str
    ) -> None:
        """Attach advanced_stats and player_stats to a single match record."""
        mid = match.get("match_id")
        is_home = match.get("team_is_home", True)
        raw_stats = stats_map.get(mid)
        match["advanced_stats"] = (
            normalize_advanced_stats(raw_stats, team_is_home=is_home)
            if raw_stats else {"warnings": ["Stats not available"]}
        )

        raw_ps = player_stats_map.get(mid)
        api_home_id = away_tid if not is_home else home_tid
        api_away_id = home_tid if not is_home else away_tid

        if raw_ps:
            match["player_stats"] = normalize_player_stats(raw_ps, api_home_id, api_away_id)
        else:
            match["player_stats"] = {"home_players": [], "away_players": [], "warnings": ["Player stats not available [N/A]"]}

    # Compute h2h advanced stats split by current home/away BEFORE enrichment,
    # since _enrich_match overwrites advanced_stats on each match.
    _h2h_adv = compute_h2h_advanced_stats(
        final["h2h"].get("matches", []),
        stats_map,
        final["match"]["home_team"],
        final["match"]["away_team"],
    )

    # Enrich all three collections with the shared helper
    for match in final["home_team_results"].get("matches", []):
        _enrich_match(match, stats_map, player_stats_map, home_team_id, away_team_id)

    for match in final["away_team_results"].get("matches", []):
        _enrich_match(match, stats_map, player_stats_map, home_team_id, away_team_id)

    for rec in final["h2h"].get("matches", []):
        _enrich_match(rec, stats_map, player_stats_map, home_team_id, away_team_id)

    # Compute advanced stats for each team (basic_stats already at top level)
    final["home_team_results"]["advanced_stats"] = compute_advanced_form(
        final["home_team_results"]["matches"]
    )
    final["away_team_results"]["advanced_stats"] = compute_advanced_form(
        final["away_team_results"]["matches"]
    )
    # h2h advanced stats already computed above with home/away split
    final["h2h"]["advanced_stats"] = {
        "home_team": _h2h_adv["home_team"],
        "away_team": _h2h_adv["away_team"],
    }

    # Aggregate player stats across all historical matches per team
    _home_ps = compute_team_player_aggregates(final["home_team_results"].get("matches", []))
    _away_ps = compute_team_player_aggregates(final["away_team_results"].get("matches", []))
    _h2h_ps = compute_team_player_aggregates(final["h2h"].get("matches", []))

    # Promote player_stats inner keys and remove wrapper
    final["home_team_results"]["player_stats_as_home"] = _home_ps.get("as_historical_home", {})
    final["home_team_results"]["player_stats_as_away"] = _home_ps.get("as_historical_away", {})
    final["away_team_results"]["player_stats_as_home"] = _away_ps.get("as_historical_home", {})
    final["away_team_results"]["player_stats_as_away"] = _away_ps.get("as_historical_away", {})
    final["h2h"]["home_team_player_stats"] = _h2h_ps.get("as_historical_home", {})
    final["h2h"]["away_team_player_stats"] = _h2h_ps.get("as_historical_away", {})

    # Remove leftover wrappers (player_stats only — form was already replaced)
    for _section in [final["home_team_results"], final["away_team_results"], final["h2h"]]:
        _section.pop("player_stats", None)

    # Strip individual match-level advanced_stats and player_stats from JSON output.
    # Aggregates (advanced_stats, player_stats_as_*) were already computed above.
    for _matches_list in [
        final["home_team_results"].get("matches", []),
        final["away_team_results"].get("matches", []),
        final["h2h"].get("matches", []),
    ]:
        for _m in _matches_list:
            _m.pop("advanced_stats", None)
            _m.pop("player_stats", None)

    # -------------------------------------------------------------------------
    # 5. LINEUPS / MISSING PLAYERS
    # -------------------------------------------------------------------------
    lineup_raw = fetch_match_lineups(event_id)
    if lineup_raw:
        final["match"]["lineups"] = normalize_lineups(lineup_raw)
    else:
        final["match"]["lineups"] = {"warnings": ["Lineup data not available [N/A]"]}

    # -------------------------------------------------------------------------
    # 6. PREVIEW (web scraping)
    # -------------------------------------------------------------------------
    home_slug = {"slug": build_preview_slug(home_url), "id": home_team_id}
    away_slug = {"slug": build_preview_slug(away_url), "id": away_team_id}
    preview_raw = fetch_preview(home_slug, away_slug, event_id)
    if preview_raw:
        final["match"]["preview"] = preview_raw
    else:
        final["match"]["preview"] = None

    # -------------------------------------------------------------------------
    # 1.5 ANOTHER MATCH DATA
    # -------------------------------------------------------------------------
    if final["match"].get("status") == "inprogress" or final["match"].get("status") == "finished":
        time.sleep(API_SLEEP_INITIAL)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(api_get, "/matches/match/summary", {"match_id": event_id}): "summary",
                executor.submit(api_get, "/matches/match/commentary", {"match_id": event_id}): "commentary",
                executor.submit(api_get, "/matches/match/stats", {"match_id": event_id}): "match_stats",
                executor.submit(api_get, "/matches/match/player-stats", {"match_id": event_id}): "player_stats",
            }
            results = {name: future.result() for future, name in futures.items()}

        final["match"]["summary"] = normalize_summary(results.get("summary"))
        final["match"]["commentary"] = normalize_commentary(results.get("commentary"))
        final["match"]["match_stats"] = normalize_match_stats(results.get("match_stats"))
        final["match"]["player_stats"] = normalize_player_stats(results.get("player_stats"), home_team_id, away_team_id)
    else:
        final["match"]["summary"] = None
        final["match"]["commentary"] = None
        final["match"]["match_stats"] = None
        final["match"]["player_stats"] = {"home_players": [], "away_players": [], "warnings": []}
        
    # -------------------------------------------------------------------------
    # 1.6 LIVE ANALYSIS DATA (inprogress only)
    # -------------------------------------------------------------------------
    final["match"]["live_analysis"] = compute_live_analysis(final["match"], final["prematch_priors"]) if final["match"].get("status") == "inprogress" else None

    # -------------------------------------------------------------------------
    # 10. TOURNAMENT STANDINGS
    # -------------------------------------------------------------------------
    if tournament_id and tournament_stage_id:
        ts_raw = fetch_tournament_standings(tournament_id, tournament_stage_id)
        if ts_raw:
            final["standings"] = normalize_standings(ts_raw, team_ids)
        else:
            final["standings"] = {"warnings": ["Tournament standings not available [N/A]"]}

    # -------------------------------------------------------------------------
    # 11. MATCH FORM STANDINGS
    # -------------------------------------------------------------------------
    form_st_raw = fetch_match_standings_form(event_id)
    if form_st_raw:
        final["form_standings"] = normalize_form_st(form_st_raw, team_ids)
    else:
        final["form_standings"] = {"warnings": ["Form standings not available [N/A]"]}

    # -------------------------------------------------------------------------
    # 12. OVER/UNDER STANDINGS
    # -------------------------------------------------------------------------
    ou_raw = fetch_match_standings_overunder(event_id)
    if ou_raw:
        final["overunder_standings"] = normalize_overunder_st(ou_raw, team_ids)
    else:
        final["overunder_standings"] = {"warnings": ["Over/Under standings not available [N/A]"]}

    # -------------------------------------------------------------------------
    # 13. MATCH TOP SCORERS
    # -------------------------------------------------------------------------
    mts_raw = fetch_match_top_scorers(event_id)
    if mts_raw:
        final["top_scorers"] = normalize_top_scorers(mts_raw, team_ids)
    else:
        final["top_scorers"] = {"warnings": ["Match top scorers not available [N/A]"]}

    # -------------------------------------------------------------------------
    # 14. TOURNAMENT TOP SCORERS
    # -------------------------------------------------------------------------
    if tournament_id and tournament_stage_id:
        tts_raw = fetch_tournament_top_scorers(tournament_id, tournament_stage_id)
        if tts_raw:
            final["tournament_top_scorers"] = normalize_top_scorers(tts_raw, team_ids)
        else:
            final["tournament_top_scorers"] = {"warnings": ["Tournament top scorers not available [N/A]"]}

    # H2H inconsistencies
    h2h_warnings = detect_h2h_inconsistencies(final["h2h"])
    final["h2h"]["warnings"].extend(h2h_warnings)

    # Cross-check warnings
    completeness = validate_data_completeness(final)
    final["home_team_results"]["warnings"].extend(completeness["home_team_results"])
    final["away_team_results"]["warnings"].extend(completeness["away_team_results"])
    final["odds"]["warnings"].extend(completeness["odds"])

    return final


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python build_match_context.py <event_id> <home_team_id> <away_team_id>", file=sys.stderr)
        sys.exit(1)

    event_id = sys.argv[1]
    home_team_id = sys.argv[2]
    away_team_id = sys.argv[3]
    
    ctx = build_context(event_id, home_team_id, away_team_id)

    # Output JSON to stdout only if connected to a TTY (not piped)
    if sys.stdout.isatty():
        print(json.dumps(ctx, indent=2, ensure_ascii=False))

    # Save JSON to ../analysis/ directory
    analysis_dir = Path(__file__).parent.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_file = analysis_dir / f"{event_id}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)

    print(f"[INFO] JSON guardado en {output_file}", file=sys.stderr)
