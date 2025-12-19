import logging
from datetime import datetime, timezone, timedelta
import pytz

# UTC compatibility for Python < 3.11
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

def setup_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def info_logger(message):
    logging.getLogger().setLevel(logging.INFO)
    logging.info(message)
    logging.getLogger().setLevel(logging.ERROR)
    


def unix_to_ist(timestamp: int) -> str:
    """
    Convert UNIX timestamp to IST local time.
    Output format: Month DD, YYYY HH:MM AM/PM IST
    """
    ist = pytz.timezone("Asia/Kolkata")

    dt_utc = datetime.fromtimestamp(timestamp, tz=UTC)
    dt_ist = dt_utc.replace(tzinfo=pytz.utc).astimezone(ist)

    return dt_ist.strftime("%B %d, %Y %I:%M %p IST")

def local_time_to_unix(
    local_time_str: str,
    offset_seconds: int = 19800,
    time_format: str = "%B %d, %Y %I:%M %p"
) -> int:
    """
    Convert formatted local time string back to UNIX timestamp.

    Example input:
      local_time_str = "December 15, 2025 02:20 PM"
      offset_seconds = 19800
    """
    local_time_str = local_time_str.strip(" IST")
    # Create timezone from offset
    tz = timezone(timedelta(seconds=offset_seconds))

    # Parse string into naive datetime
    naive_dt = datetime.strptime(local_time_str, time_format)

    # Attach timezone
    aware_dt = naive_dt.replace(tzinfo=tz)

    # Convert to UNIX timestamp
    return int(aware_dt.timestamp())
