"""Plan or explicitly execute one safely selected shadow-pass booking."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from dart.shadow_scheduler import (
    JsonlAuditStore,
    RequestsShadowBookingClient,
    ShadowScheduleConfig,
    ShadowScheduler,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("config/shadow-pass.yaml")
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-booking",
        action="store_true",
        help="explicitly authorize the external contact mutation",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.confirm_booking and not args.execute:
        raise SystemExit("--confirm-booking requires --execute")
    if args.execute and not args.confirm_booking:
        raise SystemExit("booking requires --execute --confirm-booking")
    load_dotenv()
    config = ShadowScheduleConfig.load(args.config)
    client = RequestsShadowBookingClient(
        os.getenv("KOGS_API_KEY", ""),
        config.kogs_base_url,
        config.request_timeout_s,
        config.kogs_mutation_contract_confirmed,
    )
    scheduler = ShadowScheduler(client, JsonlAuditStore(config.audit_path), config)
    now = datetime.now(UTC)
    plan = scheduler.plan(now)
    output = asdict(plan)
    if args.execute:
        output["booked_contact"] = asdict(scheduler.execute(plan, now))
    print(json.dumps(output, default=str, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
