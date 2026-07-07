#!/usr/bin/env python3
import os
import sys
import json
import time
import csv
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5

def build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = build_session()

def api_get(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: tuple[int, int] = (CONNECT_TIMEOUT, READ_TIMEOUT),
    manual_retries: int = 2,
    manual_backoff: float = 2.0,
) -> Any:
    """
    GET with:
      - requests.Session connection pooling
      - urllib3 automatic retries for transient HTTP/network errors
      - extra manual retries for read timeouts / unexpected transient failures
    """
    url = BASE_URL + path

    for attempt in range(1, manual_retries + 2):
        try:
            resp = SESSION.get(url, headers=HEADERS, params=params, timeout=timeout)

            if resp.status_code in {429, 500, 502, 503, 504}:
                print(
                    f"[WARNING] Retryable HTTP {resp.status_code} for {path} "
                    f"with params={params} | attempt {attempt}/{manual_retries + 1}",
                    file=sys.stderr,
                )
                if attempt <= manual_retries:
                    sleep_time = manual_backoff ** attempt
                    time.sleep(sleep_time)
                    continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.ReadTimeout as e:
            print(
                f"[WARNING] Read timeout for {path} with params={params} "
                f"| attempt {attempt}/{manual_retries + 1}: {e}",
                file=sys.stderr,
            )
        except requests.exceptions.ConnectTimeout as e:
            print(
                f"[WARNING] Connect timeout for {path} with params={params} "
                f"| attempt {attempt}/{manual_retries + 1}: {e}",
                file=sys.stderr,
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            print(
                f"[WARNING] HTTP error {status} for {path} with params={params} "
                f"| attempt {attempt}/{manual_retries + 1}: {e}",
                file=sys.stderr,
            )

            if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                return None

        except requests.exceptions.RequestException as e:
            print(
                f"[WARNING] API call failed for {path} with params={params} "
                f"| attempt {attempt}/{manual_retries + 1}: {e}",
                file=sys.stderr,
            )

        if attempt <= manual_retries:
            sleep_time = manual_backoff ** attempt
            time.sleep(sleep_time)

    print(f"[ERROR] API call exhausted retries for {path} with params={params}", file=sys.stderr)
    return None

# =============================================================================
# API FUNCTIONS
# =============================================================================

def fetch_tournament_standings(
    tournament_id: str,
    tournament_stage_id: str,
    stype: str = "overall"
) -> Optional[List[Dict[str, Any]]]:
    """
    GET /tournaments/standings?tournament_stage_id=...&tournament_id=...&type=...
    """
    data = api_get(
        "/tournaments/standings",
        params={
            "tournament_stage_id": tournament_stage_id,
            "tournament_id": tournament_id,
            "type": stype,
        },
    )
    return data if isinstance(data, list) else None

# =============================================================================
# CSV HELPERS
# =============================================================================

def ensure_path(path_like: Path | str) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def load_existing_values(csv_path: Path, unique_field: str) -> Set[str]:
    existing: Set[str] = set()

    if not csv_path.exists():
        return existing

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = (row.get(unique_field) or "").strip()
                if value:
                    existing.add(value)
    except Exception as e:
        print(f"[WARNING] Could not read existing CSV {csv_path}: {e}", file=sys.stderr)

    return existing


def append_rows_to_csv(
    rows: List[Dict[str, Any]],
    csv_path: Path | str,
    fieldnames: List[str],
    unique_field: str
) -> int:
    csv_path = ensure_path(csv_path)
    existing = load_existing_values(csv_path, unique_field)
    file_exists = csv_path.exists()

    rows_to_add = []
    for row in rows:
        unique_value = str(row.get(unique_field, "")).strip()
        if not unique_value or unique_value in existing:
            continue

        normalized_row = {field: row.get(field, "") for field in fieldnames}
        rows_to_add.append(normalized_row)
        existing.add(unique_value)

    if not rows_to_add:
        return 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows_to_add)

    return len(rows_to_add)

# =============================================================================
# FAILED REQUESTS
# =============================================================================

def log_failed_standings_request(
    tournament_id: str,
    tournament_stage_id: str,
    tournament_name: str,
    csv_path: Path | str
) -> None:
    csv_path = ensure_path(csv_path)

    existing: Set[tuple[str, str, str]] = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add((
                        (row.get("request_type") or "").strip(),
                        (row.get("tournament_id") or "").strip(),
                        (row.get("tournament_stage_id") or "").strip(),
                    ))
        except Exception as e:
            print(f"[WARNING] Could not read failed requests CSV {csv_path}: {e}", file=sys.stderr)

    key = ("standings", str(tournament_id).strip(), str(tournament_stage_id).strip())
    if key in existing:
        return

    file_exists = csv_path.exists()

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["request_type", "tournament_id", "tournament_stage_id", "tournament_name"]
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "request_type": "standings",
            "tournament_id": tournament_id,
            "tournament_stage_id": tournament_stage_id,
            "tournament_name": tournament_name,
        })


def retry_from_failed_csv(
    csv_path: Path | str,
    tournaments_csv: Path | str,
    teams_csv: Path | str,
    delay: float = 2.0
) -> None:
    csv_path = ensure_path(csv_path)
    teams_csv = ensure_path(teams_csv)

    if not csv_path.exists():
        print("[INFO] No failed_requests.csv found")
        return

    main_tournaments: Dict[tuple[str, str], Dict[str, str]] = {}
    if Path(tournaments_csv).exists():
        with open(tournaments_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    (row.get("tournament_id") or "").strip(),
                    (row.get("tournament_stage_id") or "").strip(),
                )
                main_tournaments[key] = row

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            request_type = (row.get("request_type") or "").strip()

            if request_type != "standings":
                continue

            tournament_id = (row.get("tournament_id") or "").strip()
            tournament_stage_id = (row.get("tournament_stage_id") or "").strip()

            if not tournament_id or not tournament_stage_id:
                print(f"[WARNING] Invalid failed row skipped: {row}", file=sys.stderr)
                continue

            print(f"[RETRY] standings tournament_id={tournament_id} stage={tournament_stage_id}")

            standings = fetch_tournament_standings(
                tournament_id=tournament_id,
                tournament_stage_id=tournament_stage_id,
                stype="overall",
            )

            if standings is None:
                print(f"[RETRY FAILED AGAIN] tournament_id={tournament_id} stage={tournament_stage_id}")
                time.sleep(delay)
                continue

            meta = main_tournaments.get((tournament_id, tournament_stage_id), {})
            team_rows = []
            for team in standings:
                team_id = str(team.get("team_id", "")).strip()
                name = str(team.get("name", "")).strip()
                if not team_id or not name:
                    continue
                team_rows.append({
                    "team_id": team_id,
                    "name": name,
                    "country_id": meta.get("country_id", ""),
                    "tournament_id": tournament_id,
                    "tournament_name": meta.get("tournament_name", ""),
                    "tournament_url": meta.get("tournament_url", ""),
                    "tournament_stage_id": tournament_stage_id,
                    "country_name": meta.get("country_name", ""),
                })

            teams_added = append_rows_to_csv(
                rows=team_rows,
                csv_path=teams_csv,
                fieldnames=[
                    "team_id",
                    "name",
                    "country_id",
                    "tournament_id",
                    "tournament_name",
                    "tournament_url",
                    "tournament_stage_id",
                    "country_name",
                ],
                unique_field="team_id",
            )

            print(f"[OK] standings {tournament_id} | teams: {len(team_rows)} | teams added: {teams_added}")
            time.sleep(delay)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def harvest_teams_from_main_tournaments(
    tournaments_csv: Path | str,
    teams_csv: Path | str,
    failed_csv: Path | str,
    request_delay: float = 0.5
) -> None:
    tournaments_csv = ensure_path(tournaments_csv)
    teams_csv = ensure_path(teams_csv)
    failed_csv = ensure_path(failed_csv)

    if not tournaments_csv.exists():
        print(f"[ERROR] File not found: {tournaments_csv}", file=sys.stderr)
        return

    tournaments: List[Dict[str, str]] = []
    with open(tournaments_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tournaments.append(row)

    print(f"[INFO] Loaded {len(tournaments)} tournaments from {tournaments_csv}")

    total_teams_added = 0

    for tournament_row in tournaments:
        tournament_id = tournament_row.get("tournament_id", "").strip()
        tournament_stage_id = tournament_row.get("tournament_stage_id", "").strip()
        tournament_name = tournament_row.get("tournament_name", "").strip()
        tournament_url = tournament_row.get("tournament_url", "").strip()
        country_id = tournament_row.get("country_id", "").strip()
        country_name = tournament_row.get("country_name", "").strip()

        if not tournament_id or not tournament_stage_id:
            print(f"[WARNING] Skipping tournament without ids: {tournament_name} ({tournament_url})", file=sys.stderr)
            continue

        print(f"[INFO] Fetching standings for: {tournament_name}")

        standings = fetch_tournament_standings(
            tournament_id=tournament_id,
            tournament_stage_id=tournament_stage_id,
            stype="overall",
        )

        if standings is None:
            print(f"[ERROR] Saving failed standings request: {tournament_name}")
            log_failed_standings_request(
                tournament_id=tournament_id,
                tournament_stage_id=tournament_stage_id,
                tournament_name=tournament_name,
                csv_path=failed_csv,
            )
            time.sleep(request_delay)
            continue

        team_rows = []
        for team in standings:
            team_id = str(team.get("team_id", "")).strip()
            name = str(team.get("name", "")).strip()

            if not team_id or not name:
                continue

            team_rows.append({
                "team_id": team_id,
                "name": name,
                "country_id": country_id,
                "tournament_id": tournament_id,
                "tournament_name": tournament_name,
                "tournament_url": tournament_url,
                "tournament_stage_id": tournament_stage_id,
                "country_name": country_name,
            })

        teams_added = append_rows_to_csv(
            rows=team_rows,
            csv_path=teams_csv,
            fieldnames=[
                "team_id",
                "name",
                "country_id",
                "tournament_id",
                "tournament_name",
                "tournament_url",
                "tournament_stage_id",
                "country_name",
            ],
            unique_field="team_id",
        )
        total_teams_added += teams_added

        print(
            f"[INFO] {tournament_name} | standings teams: {len(team_rows)} | teams added: {teams_added}"
        )

        time.sleep(request_delay)

    print("=" * 80)
    print(f"[DONE] tournaments: {tournaments_csv}")
    print(f"[DONE] teams: {teams_csv} | total new teams: {total_teams_added}")
    print(f"[DONE] failed_requests: {failed_csv}")
    print("=" * 80)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    USE_FAILED_CSV = False

    failed_csv = "failed_requests.csv"
    tournaments_csv = "main_leagues.csv"
    teams_csv = "main_teams.csv"

    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "../data"
    tournaments_csv = DATA_DIR / tournaments_csv
    failed_csv = DATA_DIR / failed_csv
    teams_csv = DATA_DIR / teams_csv
    
    if USE_FAILED_CSV:
        print("[MODE] RETRY FROM CSV")
        retry_from_failed_csv(
            csv_path=failed_csv,
            tournaments_csv=tournaments_csv,
            teams_csv=teams_csv,
            delay=2.0,
        )
        sys.exit(0)

    harvest_teams_from_main_tournaments(
        tournaments_csv=tournaments_csv,
        teams_csv=teams_csv,
        failed_csv=failed_csv,
        request_delay=0.5,
    )
