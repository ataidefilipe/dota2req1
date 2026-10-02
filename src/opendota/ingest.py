import logging
import argparse
import sys
from .client import OpenDotaClient
from ..db import repo
from ..config import Config
from ..features.filters import is_valid_match

logger = logging.getLogger(__name__)

def fetch_and_save_match(client: OpenDotaClient, match_id: int, dataset: str, target_patch: int = 59, max_leaver_status: int = 0) -> bool:
    """
    Fetches full match details, validates quality filters,
    and atomically saves match, draft_events, and match_players.
    """
    try:
        data = client.get(f"/matches/{match_id}")
        if not data:
            logger.warning(f"Match {match_id} not found or empty.")
            return False

        # Apply Quality Filters (Captain's Mode, 10 picks, 5/side, duration >= 30m, leavers <= max_leaver_status, Patch 59)
        if not is_valid_match(data, target_patch=target_patch, max_leaver_status=max_leaver_status):
            logger.debug(f"Match {match_id} filtered out (criteria not met).")
            return False

        # Atomic persistence
        success = repo.save_full_match(data, dataset)
        return success
    except Exception as e:
        logger.error(f"Failed to ingest match {match_id}: {e}")
        return False

def ingest_pro_matches(limit: int = 100, direction: str = "forward", max_scanned: int = 5000):
    client = OpenDotaClient()
    dataset = "pro"
    
    matches_saved = 0
    total_scanned = 0
    
    if direction == "backfill":
        # Backfill mode: crawl backwards in history
        last_min_id = repo.get_ingestion_state("pro_backfill")
        if last_min_id == 0:
            with repo.get_db_connection() as conn:
                row = conn.execute("SELECT min(match_id) FROM matches WHERE dataset = 'pro'").fetchone()
                last_min_id = row[0] if (row and row[0]) else 0
        
        logger.info(f"Iniciando Backfill de Pro Matches. Ponto de partida (less_than_match_id): {last_min_id or 'Mais recentes'}")
        current_cursor = last_min_id if last_min_id > 0 else None

        try:
            while matches_saved < limit and total_scanned < max_scanned:
                params = {}
                if current_cursor:
                    params["less_than_match_id"] = current_cursor

                batch = client.get("/proMatches", params)
                if not batch:
                    logger.info("Nenhuma partida adicional retornada pela API.")
                    break

                logger.info(f"Recebida página com {len(batch)} partidas pro. Processando...")

                for m in batch:
                    mid = m["match_id"]
                    current_cursor = mid if current_cursor is None else min(current_cursor, mid)
                    total_scanned += 1

                    if fetch_and_save_match(client, mid, dataset):
                        matches_saved += 1
                        logger.info(f"[+] Match {mid} salvo com sucesso! (Salvas: {matches_saved}/{limit} | Escaneadas: {total_scanned})")
                    else:
                        logger.debug(f"[-] Match {mid} descartado.")

                    if matches_saved >= limit:
                        break

                # Salva checkpoint do cursor mais antigo após cada lote
                if current_cursor:
                    repo.set_ingestion_state("pro_backfill", current_cursor)

        except KeyboardInterrupt:
            logger.warning("Interrupção pelo usuário recebida. Salvando estado atual...")
        finally:
            if current_cursor:
                repo.set_ingestion_state("pro_backfill", current_cursor)
            logger.info(f"Backfill finalizado. Total salvo nesta sessão: {matches_saved} (Total escaneado: {total_scanned})")

    else:
        # Forward mode: crawl forward for fresh matches
        last_id = repo.get_ingestion_state("pro")
        if last_id == 0:
            with repo.get_db_connection() as conn:
                row = conn.execute("SELECT max(match_id) FROM matches WHERE dataset = 'pro'").fetchone()
                last_id = row[0] if (row and row[0]) else 0

        logger.info(f"Iniciando Ingestão Incremental (Forward). Último match_id conhecido: {last_id}")
        max_id_seen = last_id

        try:
            # Pega as partidas mais recentes
            batch = client.get("/proMatches", {})
            if not batch:
                logger.info("Nenhuma partida encontrada.")
                return

            # Ordena por match_id ASC para processar cronologicamente
            batch.sort(key=lambda x: x["match_id"])
            new_matches = [m for m in batch if m["match_id"] > last_id]

            logger.info(f"Encontradas {len(new_matches)} novas partidas pro desde o último checkpoint.")

            for m in new_matches:
                mid = m["match_id"]
                total_scanned += 1
                logger.info(f"Processando Pro Match {mid}...")

                if fetch_and_save_match(client, mid, dataset):
                    matches_saved += 1
                    logger.info(f"[+] Match {mid} salvo! (Salvas: {matches_saved}/{limit})")

                if mid > max_id_seen:
                    max_id_seen = mid
                    repo.set_ingestion_state("pro", max_id_seen)

                if matches_saved >= limit:
                    logger.info("Limite atingido.")
                    break

        except KeyboardInterrupt:
            logger.warning("Interrupção pelo usuário. Atualizando checkpoint...")
        finally:
            if max_id_seen > last_id:
                repo.set_ingestion_state("pro", max_id_seen)
            logger.info(f"Ingestão Forward concluída. Total salvo: {matches_saved}")

def ingest_public_matches(limit: int = 100, direction: str = "backfill", max_scanned: int = 5000):
    client = OpenDotaClient()
    dataset = "ranked_high"
    min_rank = Config.PUBLICMATCHES_MIN_RANK

    logger.warning(
        "Atenção: Partidas públicas ranqueadas geralmente utilizam All Pick (game_mode=22). "
        "Sob o filtro de Captain's Mode (game_mode=2), a maioria absoluta será rejeitada."
    )

    last_id = repo.get_ingestion_state("ranked_high_backfill")
    current_cursor = last_id if last_id > 0 else None
    matches_saved = 0
    total_scanned = 0

    try:
        while matches_saved < limit and total_scanned < max_scanned:
            params = {"min_rank": min_rank}
            if current_cursor:
                params["less_than_match_id"] = current_cursor

            batch = client.get("/publicMatches", params)
            if not batch:
                logger.info("Nenhuma partida pública retornada.")
                break

            for m in batch:
                mid = m["match_id"]
                current_cursor = mid if current_cursor is None else min(current_cursor, mid)
                total_scanned += 1

                if fetch_and_save_match(client, mid, dataset):
                    matches_saved += 1
                    logger.info(f"[+] Match público {mid} salvo! ({matches_saved}/{limit})")

                if matches_saved >= limit:
                    break

            if current_cursor:
                repo.set_ingestion_state("ranked_high_backfill", current_cursor)

    except KeyboardInterrupt:
        logger.warning("Interrupção pelo usuário recebida.")
    finally:
        if current_cursor:
            repo.set_ingestion_state("ranked_high_backfill", current_cursor)
        logger.info(f"Ingestão pública concluída. Salvas: {matches_saved} (Escaneadas: {total_scanned})")

def ingest_patch59_pro_window(limit: int = 600):
    """
    Ingere cirurgicamente todas as partidas profissionais do Patch 59 
    compreendidas entre 20/02/2026 e 24/03/2026 via OpenDota Explorer SQL.
    """
    client = OpenDotaClient()
    dataset = "pro"
    
    sql = """
    SELECT m.match_id
    FROM matches m
    JOIN leagues l ON m.leagueid = l.leagueid
    WHERE m.game_mode = 2 
      AND l.tier in ('professional', 'premium')
      AND m.start_time >= 1771620603 
      AND m.start_time < 1774313459
    ORDER BY m.match_id ASC
    """
    logger.info("Consultando lista de partidas profissionais do Patch 59 via OpenDota Explorer...")
    res = client.get("/explorer", params={"sql": sql})
    if not res or "rows" not in res:
        logger.error("Falha ao obter lista de partidas do Explorer.")
        return

    mids = [row["match_id"] for row in res.get("rows", [])]
    logger.info(f"Total de {len(mids)} partidas profissionais encontradas no período restante do Patch 59.")

    with repo.get_db_connection() as conn:
        if mids:
            placeholders = ','.join('?' for _ in mids)
            existing = {r[0] for r in conn.execute(f"SELECT match_id FROM matches WHERE match_id IN ({placeholders})", mids).fetchall()}
        else:
            existing = set()

    to_fetch = [mid for mid in mids if mid not in existing]
    logger.info(f"{len(existing)} partidas já constam no banco. Restam {len(to_fetch)} para processar.")

    matches_saved = 0
    total_scanned = 0

    try:
        for mid in to_fetch:
            total_scanned += 1
            if fetch_and_save_match(client, mid, dataset, target_patch=59):
                matches_saved += 1
                logger.info(f"[+] Match {mid} salvo com sucesso! (Salvas: {matches_saved} | Escaneadas: {total_scanned}/{len(to_fetch)})")
            else:
                logger.debug(f"[-] Match {mid} descartado pelos filtros.")

            if matches_saved >= limit:
                logger.info("Limite de partidas atingido.")
                break
    except KeyboardInterrupt:
        logger.warning("Interrupção pelo usuário recebida.")
    finally:
        logger.info(f"Conclusão do Patch 59 finalizada. Total salvo: {matches_saved} (Escaneadas: {total_scanned})")

def ingest_patch59_community_matches(limit: int = 500):
    """
    Ingere partidas de Captain's Mode de ligas comunitárias, amadoras e semiprofissionais
    (tier = 'excluded') pertencentes ao Patch 59 (Dota 2 v7.40) via OpenDota Explorer SQL.
    Aplica pré-filtro m.duration >= 1800 diretamente no SQL para economizar requisições à API.
    """
    client = OpenDotaClient()
    dataset = "community"
    
    sql = """
    SELECT m.match_id
    FROM matches m
    JOIN leagues l ON m.leagueid = l.leagueid
    WHERE m.game_mode = 2 
      AND l.tier = 'excluded'
      AND m.duration >= 1800
      AND m.start_time >= 1765849783 
      AND m.start_time < 1774313459
    ORDER BY m.match_id ASC
    """
    logger.info("Consultando lista de partidas comunitárias/amadoras do Patch 59 via OpenDota Explorer...")
    res = client.get("/explorer", params={"sql": sql})
    if not res or "rows" not in res:
        logger.error("Falha ao obter lista de partidas comunitárias do Explorer.")
        return

    mids = [row["match_id"] for row in res.get("rows", [])]
    logger.info(f"Total de {len(mids)} partidas comunitárias (duration >= 30m) encontradas no Patch 59.")

    with repo.get_db_connection() as conn:
        if mids:
            placeholders = ','.join('?' for _ in mids)
            existing = {r[0] for r in conn.execute(f"SELECT match_id FROM matches WHERE match_id IN ({placeholders})", mids).fetchall()}
        else:
            existing = set()

    to_fetch = [mid for mid in mids if mid not in existing]
    logger.info(f"{len(existing)} partidas já constam no banco. Restam {len(to_fetch)} para processar.")

    matches_saved = 0
    total_scanned = 0

    try:
        for mid in to_fetch:
            total_scanned += 1
            if fetch_and_save_match(client, mid, dataset, target_patch=59, max_leaver_status=1):
                matches_saved += 1
                logger.info(f"[+] Match comunitário {mid} salvo com sucesso! (Salvas: {matches_saved}/{limit} | Escaneadas: {total_scanned}/{len(to_fetch)})")
            else:
                logger.debug(f"[-] Match comunitário {mid} descartado pelos filtros.")

            if matches_saved >= limit:
                logger.info("Limite de partidas atingido.")
                break
    except KeyboardInterrupt:
        logger.warning("Interrupção pelo usuário recebida.")
    finally:
        logger.info(f"Ingestão comunitária do Patch 59 finalizada. Total salvo: {matches_saved} (Escaneadas: {total_scanned})")

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Ingestão contínua de partidas OpenDota")
    parser.add_argument("--complete-patch59", action="store_true", help="Completar 100% das partidas pro do Patch 59")
    parser.add_argument("--community-patch59", action="store_true", help="Ingerir partidas comunitárias/amadoras do Patch 59")
    parser.add_argument("--dataset", choices=["pro", "ranked_high", "community"], default="pro", help="Origem do dataset")
    parser.add_argument("--direction", choices=["forward", "backfill"], default="forward", help="Sentido da coleta")
    parser.add_argument("--limit", type=int, default=500, help="Quantidade de partidas aceitas a salvar")
    parser.add_argument("--max-scanned", type=int, default=2000, help="Limite máximo de partidas a inspecionar")
    args = parser.parse_args()

    if args.complete_patch59:
        ingest_patch59_pro_window(limit=args.limit)
    elif args.community_patch59 or args.dataset == "community":
        ingest_patch59_community_matches(limit=args.limit)
    elif args.dataset == "pro":
        ingest_pro_matches(limit=args.limit, direction=args.direction, max_scanned=args.max_scanned)
    elif args.dataset == "ranked_high":
        ingest_public_matches(limit=args.limit, direction=args.direction, max_scanned=args.max_scanned)
