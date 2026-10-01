import time
import requests
import logging
from ..config import Config

class OpenDotaClient:
    BASE_URL = "https://api.opendota.com/api"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.last_request_time = 0
        self.rps = Config.OPENDOTA_RPS
        self.api_key = Config.OPENDOTA_API_KEY
        
        self.min_interval = 1.0 / self.rps if self.rps > 0 else 0
        
    def _throttle(self):
        """Sleeps if necessary to respect RPS."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def get(self, endpoint: str, params: dict = None, retries: int = 5):
        """
        Executes a GET request with throttling and retries.
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        if self.api_key:
            params['api_key'] = self.api_key
            
        for attempt in range(retries):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=15)
                
                # Rate limit hit
                if response.status_code == 429:
                    resp_text = response.text.lower()
                    rem_day = response.headers.get("X-Rate-Limit-Remaining-Day")
                    if "daily api limit exceeded" in resp_text or (rem_day is not None and int(rem_day) <= 0):
                        self.logger.error(
                            "Limite diário da API OpenDota atingido (2.000 req/dia no IP gratuito). "
                            "O limite reseta às 00:00 UTC (21:00 BRT) ou insira uma OPENDOTA_API_KEY no arquivo .env."
                        )
                        return None

                    wait = int(response.headers.get("Retry-After", 5))
                    self.logger.warning(f"429 Rate Limit por minuto. Aguardando {wait}s...")
                    time.sleep(wait)
                    continue
                
                # Server errors (500, 502, 503, 504)
                if 500 <= response.status_code < 600:
                    wait_time = min(2 ** (attempt + 1), 30)
                    self.logger.warning(f"Server error {response.status_code} on {url}. Retry {attempt+1}/{retries} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                self.logger.error(f"Request failed: {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(min(2 ** attempt, 10))

        
        return None
