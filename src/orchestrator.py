"""
FarmSignal AI — automation orchestrator.

This is the piece that distinguishes an AI *engineering* project from a
notebook: it watches risk scores and, without a human reviewing each case,
decides who gets contacted, how, and logs every decision for audit and
monitoring.

Design intent: the SMS/CRM clients below are mocked (they print + log
instead of calling a real API), but they are written as drop-in interfaces.
Swapping MockSMSClient for a real Africa's Talking client, or MockCRMClient
for a real CRM's REST API, requires no change to the rules engine itself —
that separation is the actual "integration architecture" this role expects.
"""

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCORES_PATH = "data/risk_scores.csv"
LOG_PATH = "logs/actions_log.csv"

RNG = random.Random(7)


# ---------------------------------------------------------------------------
# Channel clients — mocked, but built to the shape of a real integration.
# In production, MockSMSClient.send() becomes a POST to Africa's Talking's
# /messaging endpoint; MockCRMClient.create_task() becomes a POST to
# whatever CRM/loan-management system holds the field-agent task queue.
# ---------------------------------------------------------------------------

class MockSMSClient:
    def send(self, farmer_id: str, message: str) -> dict:
        # Real implementation: requests.post(AT_SMS_URL, data={...}, headers={"apikey": AT_API_KEY})
        delivered = RNG.random() > 0.05  # simulate ~95% delivery success
        return {"channel": "sms", "status": "delivered" if delivered else "failed", "detail": message}


class MockCRMClient:
    def create_task(self, farmer_id: str, task_type: str, priority: str) -> dict:
        # Real implementation: requests.post(CRM_TASKS_URL, json={...}, headers={"Authorization": ...})
        return {"channel": "crm_task", "status": "created", "detail": f"{task_type} ({priority})"}


class MockEscalationClient:
    def escalate(self, farmer_id: str, reason: str) -> dict:
        # Real implementation: notify a supervisor via Slack/Teams webhook or email API
        return {"channel": "escalation", "status": "sent", "detail": reason}


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    farmer_id: str
    risk_tier: str
    risk_score: float
    action_type: str
    channel: str
    status: str
    detail: str
    timestamp: str


def decide_and_act(row: pd.Series, sms: MockSMSClient, crm: MockCRMClient,
                    escalation: MockEscalationClient) -> ActionResult:
    """
    Core orchestration rule. This is intentionally simple and readable —
    in production this is where you'd add cooldown windows (don't nudge the
    same farmer twice in 7 days), channel fallback (SMS fails -> try CRM
    task), and multi-armed-bandit testing of message wording.
    """
    farmer_id = row["farmer_id"]
    tier = row["risk_tier"]
    reasons = row["top_reasons"]

    if tier == "high":
        result = escalation.escalate(
            farmer_id,
            reason=f"High default risk (score={row['risk_score']:.2f}) - {reasons}",
        )
        action_type = "supervisor_escalation"

    elif tier == "medium":
        result = crm.create_task(
            farmer_id, task_type="Field visit follow-up", priority="medium"
        )
        action_type = "field_agent_task"

    else:
        message = "Reminder: your next input redemption window is open. Visit your agrodealer this week."
        result = sms.send(farmer_id, message)
        action_type = "sms_nudge" if tier == "low" else "no_action"

    return ActionResult(
        farmer_id=farmer_id,
        risk_tier=tier,
        risk_score=row["risk_score"],
        action_type=action_type,
        channel=result["channel"],
        status=result["status"],
        detail=result["detail"],
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def run():
    scores = pd.read_csv(SCORES_PATH, encoding="utf-8")
    sms, crm, escalation = MockSMSClient(), MockCRMClient(), MockEscalationClient()

    results = [decide_and_act(row, sms, crm, escalation) for _, row in scores.iterrows()]

    Path("logs").mkdir(exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(vars(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(vars(r))

    log_df = pd.DataFrame([vars(r) for r in results])
    print(f"Processed {len(log_df)} farmers -> {LOG_PATH}")
    print(log_df.groupby(["risk_tier", "action_type", "status"]).size())


if __name__ == "__main__":
    run()
