import { api } from "./client";
import type { Alert, AlertRun } from "../types";

export function getAlerts(includeAcknowledged = false): Promise<Alert[]> {
  return api.get<Alert[]>(`/api/alerts?include_acknowledged=${includeAcknowledged}`);
}

/** One local generation per holding: minutes for a large portfolio. */
export function runAlerts(ticker?: string): Promise<AlertRun> {
  return api.post<AlertRun>(`/api/alerts/run${ticker ? `?ticker=${encodeURIComponent(ticker)}` : ""}`);
}

export function acknowledgeAlert(id: number): Promise<Alert> {
  return api.post<Alert>(`/api/alerts/${id}/acknowledge`);
}
