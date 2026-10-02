import logging

logger = logging.getLogger(__name__)

def is_valid_match(match_data: dict, target_patch: int = None, max_leaver_status: int = 0) -> bool:
    """
    Validates if a match meets the strict criteria for the dataset:
    1. Game Mode == 2 (Captain's Mode)
    2. Duration >= 1800s (30 mins)
    3. Target radiant_win is boolean (not None)
    4. Picks/Bans: Exactly 10 picks total, 5 per team, no duplicate picked heroes
    5. Players: Exactly 10 players, no abandons (leaver_status <= max_leaver_status), 5 per team
    6. Patch == target_patch (if target_patch is specified and patch is present)
    """
    mid = match_data.get('match_id')

    # 0. Patch filter
    if target_patch is not None:
        p = match_data.get('patch')
        if p is not None and p != target_patch:
            logger.debug(f"Match {mid} rejected: patch {p} != {target_patch}")
            return False

    # 1. Game Mode
    if match_data.get('game_mode') != 2:
        logger.debug(f"Match {mid} rejected: game_mode {match_data.get('game_mode')} != 2 (Captain's Mode)")
        return False

    # 2. Duration
    duration = match_data.get('duration', 0)
    if duration < 1800:
        logger.debug(f"Match {mid} rejected: duration {duration} < 1800")
        return False

    # 3. Target
    if match_data.get('radiant_win') is None:
        logger.debug(f"Match {mid} rejected: radiant_win is None")
        return False

    # 4. Draft Completeness & Integrity
    pbs = match_data.get('picks_bans')
    if not pbs:
        logger.debug(f"Match {mid} rejected: no picks_bans")
        return False

    picks = [x for x in pbs if x.get('is_pick')]
    if len(picks) != 10:
        logger.debug(f"Match {mid} rejected: {len(picks)} picks (expected exactly 10)")
        return False

    # Check team distribution if team is specified
    rad_picks = [p for p in picks if p.get('team') == 0]
    dire_picks = [p for p in picks if p.get('team') == 1]
    if (rad_picks or dire_picks) and (len(rad_picks) != 5 or len(dire_picks) != 5):
        logger.debug(f"Match {mid} rejected: unbalanced picks (Radiant: {len(rad_picks)}, Dire: {len(dire_picks)})")
        return False

    # Check hero uniqueness if hero_id is specified
    hero_ids = [p.get('hero_id') for p in picks if p.get('hero_id') is not None]
    if hero_ids and len(set(hero_ids)) != len(hero_ids):
        logger.debug(f"Match {mid} rejected: duplicate hero in picks")
        return False

    # 5. Players & Leaver Status
    players = match_data.get('players', [])
    if not players:
        logger.debug(f"Match {mid} rejected: no players data")
        return False

    if len(players) != 10:
        logger.debug(f"Match {mid} rejected: {len(players)} players (expected 10)")
        return False

    # Check leaver status (0: finished normally, 1: disconnected at end/GG, >=2: actual abandon/afk)
    for p in players:
        if p.get('leaver_status', 0) > max_leaver_status:
            logger.debug(f"Match {mid} rejected: player {p.get('player_slot')} has leaver_status {p.get('leaver_status')} > {max_leaver_status}")
            return False

    # Check player slot distribution if player_slot is present
    slots = [p.get('player_slot') for p in players if p.get('player_slot') is not None]
    if len(slots) == 10:
        if len(set(slots)) != 10:
            logger.debug(f"Match {mid} rejected: duplicate player_slots")
            return False
        rad_players = [s for s in slots if s < 128]
        dire_players = [s for s in slots if s >= 128]
        if len(rad_players) != 5 or len(dire_players) != 5:
            logger.debug(f"Match {mid} rejected: unbalanced player slots (Radiant: {len(rad_players)}, Dire: {len(dire_players)})")
            return False

    return True
