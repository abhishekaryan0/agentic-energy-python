"""Runtime configuration for Snowflake-to-S3 parquet transformations."""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ORG_ID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

DEFAULT_TABLES = ("well_sensor_readings", "well_daily_production")
TRANSFORM_MODES = ("full", "incremental")


def _strip(value: Optional[str]) -> str:
    return (value or "").strip()


def _split_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_organization_id_list(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [_strip(str(item)) for item in raw if _strip(str(item))]
    return [_strip(part) for part in str(raw).split(",") if _strip(part)]


def load_organization_ids(overrides: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return tenant UUIDs from ORGANIZATION_IDS (comma-separated) or ORGANIZATION_ID."""
    overrides = overrides or {}
    raw_list = overrides.get("organization_ids")
    if raw_list is None:
        raw_list = os.getenv("ORGANIZATION_IDS")
    if raw_list:
        ids = _parse_organization_id_list(raw_list)
        if not ids:
            raise ValueError("ORGANIZATION_IDS is empty")
        return ids

    single = _strip(overrides.get("organization_id") or os.getenv("ORGANIZATION_ID"))
    if single:
        return [single]
    raise ValueError("ORGANIZATION_ID or ORGANIZATION_IDS is required")


def resolve_organization_ids(event: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Resolve which organizations to export for a handler/CLI/Lambda event.

    Priority: event organization_id -> event organization_ids -> env ORGANIZATION_IDS -> ORGANIZATION_ID.
    """
    event = event or {}

    single = _strip(event.get("organization_id"))
    if single:
        return [single]

    raw_list = event.get("organization_ids")
    if raw_list:
        ids = _parse_organization_id_list(raw_list)
        if not ids:
            raise ValueError("organization_ids is empty")
        return ids

    return load_organization_ids()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class SnowflakeConnectionConfig:
    """Snowflake connection values sourced from SNOWFLAKE_* environment variables."""

    user: Optional[str]
    password: Optional[str]
    account: Optional[str]
    warehouse: Optional[str]
    database: Optional[str]
    schema: Optional[str]
    role: Optional[str] = None
    token: Optional[str] = None
    passcode: Optional[str] = None
    authenticator: Optional[str] = None
    network_policy: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SnowflakeConnectionConfig":
        token = (os.getenv("SNOWFLAKE_TOKEN") or "").strip() or None
        return cls(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE", "AT_RISK_ASSETS"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "SENSOR_DATA"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            token=token,
            passcode=(os.getenv("SNOWFLAKE_PASSCODE") or "").strip() or None,
            authenticator=(os.getenv("SNOWFLAKE_AUTHENTICATOR") or "").strip() or None,
            network_policy=(os.getenv("SNOWFLAKE_NETWORK_POLICY") or "").strip() or None,
        )

    def validate(self) -> None:
        missing = [
            name
            for name in ("user", "account", "warehouse", "database", "schema", "role")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"Missing Snowflake env values: {', '.join(missing)}")

    def as_connect_kwargs(self) -> Dict[str, Any]:
        self.validate()
        kwargs: Dict[str, Any] = {
            "user": self.user,
            "account": self.account,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "role": self.role,
        }
        if self.network_policy:
            kwargs["network_policy"] = self.network_policy
        token = (self.token or "").strip()
        auth = (self.authenticator or "").strip()
        use_password = os.getenv("SNOWFLAKE_USE_PASSWORD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        passcode = (self.passcode or "").strip()

        if token and (not use_password):
            # Snowflake PAT: pass token as password (default authenticator).
            kwargs["password"] = token
            return {k: v for k, v in kwargs.items() if v is not None and v != ""}

        if auth and not use_password:
            kwargs["authenticator"] = auth
            if self.password:
                kwargs["password"] = self.password
            return {k: v for k, v in kwargs.items() if v is not None and v != ""}

        if not self.password:
            raise ValueError(
                "Snowflake auth: set SNOWFLAKE_TOKEN (PAT), or SNOWFLAKE_PASSWORD "
                "(use SNOWFLAKE_USE_PASSWORD=1 to force password when a token is also set)."
            )
        kwargs["password"] = self.password
        if passcode:
            kwargs["passcode"] = passcode
        return {k: v for k, v in kwargs.items() if v is not None and v != ""}


@dataclass
class TransformationConfig:
    """Configuration for exporting Snowflake tables into S3 parquet datasets."""

    snowflake: SnowflakeConnectionConfig
    s3_bucket: str
    organization_id: Optional[str] = None
    output_prefix: Optional[str] = None
    tables: List[str] = field(default_factory=lambda: list(DEFAULT_TABLES))
    batch_size: int = 10000
    write_empty_files: bool = True
    aws_region: Optional[str] = None
    mode: str = "full"
    write_partitions: bool = True
    write_snapshots: bool = False
    merge_snapshots: bool = False

    @classmethod
    def from_env(cls) -> "TransformationConfig":
        tables = _split_csv(os.getenv("TRANSFORM_TABLES", "")) or list(DEFAULT_TABLES)
        output_prefix = (
            os.getenv("TRANSFORM_OUTPUT_S3_PREFIX")
            or os.getenv("TRANSFORMATION_OUTPUT_S3_PREFIX")
            or None
        )
        mode = (os.getenv("TRANSFORM_MODE", "full") or "full").strip().lower()
        return cls(
            snowflake=SnowflakeConnectionConfig.from_env(),
            s3_bucket=os.getenv("S3_BUCKET", ""),
            organization_id=os.getenv("ORGANIZATION_ID"),
            output_prefix=output_prefix,
            tables=tables,
            batch_size=int(os.getenv("TRANSFORM_BATCH_SIZE", "10000") or "10000"),
            write_empty_files=_env_bool("TRANSFORM_WRITE_EMPTY_FILES", True),
            aws_region=os.getenv("AWS_REGION") or None,
            mode=mode,
            write_partitions=_env_bool("TRANSFORM_WRITE_PARTITIONS", True),
            write_snapshots=_env_bool("TRANSFORM_WRITE_SNAPSHOTS", False),
            merge_snapshots=_env_bool("TRANSFORM_MERGE_SNAPSHOTS", False),
        )

    def validate(self) -> None:
        self.snowflake.validate()
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET is required")
        if self.batch_size < 1:
            raise ValueError("TRANSFORM_BATCH_SIZE must be >= 1")
        if not self.write_partitions and not self.write_snapshots:
            raise ValueError(
                "At least one of partition or snapshot output must be enabled "
                "(TRANSFORM_WRITE_PARTITIONS or TRANSFORM_WRITE_SNAPSHOTS)."
            )
        if self.mode not in TRANSFORM_MODES:
            raise ValueError(f"TRANSFORM_MODE must be one of {TRANSFORM_MODES}")
        if self.mode == "incremental" and not self.organization_id:
            raise ValueError("ORGANIZATION_ID is required for incremental transform mode")
        if not self.organization_id and not _env_bool(
            "TRANSFORM_ALLOW_CROSS_TENANT_EXPORT", False
        ):
            raise ValueError(
                "ORGANIZATION_ID is required: exports use "
                'WHERE "organization_id" = %s in Snowflake (same tenant column as RED_BUFF) '
                "and write to s3://$S3_BUCKET/{ORGANIZATION_ID}/sensor_data/. "
                "Set TRANSFORM_ALLOW_CROSS_TENANT_EXPORT=1 only for intentional full-table exports."
            )

    def sensor_data_prefix(self) -> str:
        """Return the S3 prefix used by downstream ML packages for source parquet."""
        if self.output_prefix:
            return self.output_prefix.strip().strip("/")
        if self.organization_id:
            return f"{self.organization_id}/sensor_data"
        return "sensor_data"

    def with_org_snowflake_schema(self) -> "TransformationConfig":
        """Return a copy using the Snowflake schema for this tenant (if organization_id set)."""
        if not self.organization_id:
            return self
        from transformations.org_snowflake import snowflake_connection_for_org

        sf = snowflake_connection_for_org(self.snowflake, self.organization_id)
        if sf is self.snowflake:
            return self
        updated = TransformationConfig(
            snowflake=sf,
            s3_bucket=self.s3_bucket,
            organization_id=self.organization_id,
            output_prefix=self.output_prefix,
            tables=list(self.tables),
            batch_size=self.batch_size,
            write_empty_files=self.write_empty_files,
            aws_region=self.aws_region,
            mode=self.mode,
            write_partitions=self.write_partitions,
            write_snapshots=self.write_snapshots,
            merge_snapshots=self.merge_snapshots,
        )
        return updated
