"""Well operational status (RUNNING / IDLE / SHUTDOWN) from sensor readings."""

from well_status.service import update_all_well_statuses

__all__ = ["update_all_well_statuses"]
