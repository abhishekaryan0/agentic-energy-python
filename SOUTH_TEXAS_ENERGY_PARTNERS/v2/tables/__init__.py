"""One module per STEP V2 Snowflake table (DDL + columns + row mappers)."""

from .synth_asset_daily_history_v2 import SynthAssetDailyHistoryV2
from .synth_assets_v2 import SynthAssetsV2
from .synth_invoices_v2 import SynthInvoicesV2
from .synth_personnel_v2 import SynthPersonnelV2
from .synth_pipeline_state_v2 import SynthPipelineStateV2
from .synth_revenue_v2 import SynthRevenueV2
from .well_daily_production_v2 import WellDailyProductionV2
from .well_sensor_readings_v2 import WellSensorReadingsV2

ALL_TABLES = (
    WellDailyProductionV2,
    WellSensorReadingsV2,
    SynthAssetsV2,
    SynthPersonnelV2,
    SynthAssetDailyHistoryV2,
    SynthInvoicesV2,
    SynthRevenueV2,
    SynthPipelineStateV2,
)

__all__ = [
    "ALL_TABLES",
    "WellDailyProductionV2",
    "WellSensorReadingsV2",
    "SynthAssetsV2",
    "SynthPersonnelV2",
    "SynthAssetDailyHistoryV2",
    "SynthInvoicesV2",
    "SynthRevenueV2",
    "SynthPipelineStateV2",
]
