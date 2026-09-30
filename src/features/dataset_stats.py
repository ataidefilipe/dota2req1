import logging
import json
import os
import sqlite3
from datetime import datetime
from ..db.repo import get_db_connection
from ..config import Config

logger = logging.getLogger(__name__)

def generate_stats(output_dir='reports') -> dict:
    stats = {}

    with get_db_connection() as conn:
        # 1. Total matches per dataset
        cursor = conn.execute("SELECT dataset, COUNT(*) FROM matches GROUP BY dataset")
        stats['matches_by_dataset'] = dict(cursor.fetchall())

        # 2. Win rate by side (Radiant)
        cursor = conn.execute("SELECT avg(radiant_win) FROM matches")
        row = cursor.fetchone()
        stats['radiant_win_rate'] = float(row[0]) if (row and row[0] is not None) else 0.0

        # 3. Total ingested matches
        cursor = conn.execute("SELECT COUNT(*) FROM matches")
        stats['total_matches'] = int(cursor.fetchone()[0])

        # 4. Duration statistics (minutos)
        cursor = conn.execute("SELECT min(duration), avg(duration), max(duration) FROM matches")
        d_min, d_avg, d_max = cursor.fetchone()
        stats['duration_minutes'] = {
            'min': round(d_min / 60.0, 2) if d_min else 0.0,
            'avg': round(d_avg / 60.0, 2) if d_avg else 0.0,
            'max': round(d_max / 60.0, 2) if d_max else 0.0
        }

        # 5. Distribution by Patch
        cursor = conn.execute("SELECT patch_id, COUNT(*) FROM matches GROUP BY patch_id ORDER BY COUNT(*) DESC")
        stats['matches_by_patch'] = {str(r[0]): r[1] for r in cursor.fetchall()}

        # 6. Player skill coverage (rank_tier & computed_mmr)
        cursor = conn.execute("""
            SELECT 
                COUNT(*),
                COUNT(rank_tier),
                COUNT(computed_mmr),
                AVG(computed_mmr)
            FROM match_players
        """)
        p_total, p_rank, p_mmr, p_mmr_avg = cursor.fetchone()
        stats['player_telemetry'] = {
            'total_players_recorded': p_total or 0,
            'players_with_rank_tier': p_rank or 0,
            'rank_tier_coverage_pct': round((p_rank / p_total * 100), 2) if p_total else 0.0,
            'players_with_computed_mmr': p_mmr or 0,
            'computed_mmr_coverage_pct': round((p_mmr / p_total * 100), 2) if p_total else 0.0,
            'avg_computed_mmr': round(p_mmr_avg, 2) if p_mmr_avg else 0.0
        }

        # 7. Date range
        cursor = conn.execute("SELECT min(start_time), max(start_time) FROM matches")
        mn, mx = cursor.fetchone()
        from datetime import timezone
        stats['time_range'] = {
            'start_timestamp': mn,
            'start_utc': datetime.fromtimestamp(mn, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if mn else None,
            'end_timestamp': mx,
            'end_utc': datetime.fromtimestamp(mx, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if mx else None
        }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'dataset_stats.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Relatório de telemetria gerado em {path}")
    print(json.dumps(stats, indent=2))
    return stats

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    generate_stats()
