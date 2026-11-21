// Host-side binding for user association decisions. Actor IDs are validated
// again by the daemon; this guard pins the app to its acknowledged open case.
export class InvestigationAppAccess {
  private epoch = 0;
  private caseId = "";
  private revision = 0;

  begin(name: string): number {
    if (name === "investigate.case.open") { ++this.epoch; this.caseId = ""; this.revision = 0; }
    return this.epoch;
  }

  authorize(name: string, args: Record<string, unknown>): void {
    if (name !== "investigate.association.review") return;
    if (!this.caseId || args.case_id !== this.caseId || args.expected_revision !== this.revision)
      throw new Error("Open the current case revision before reviewing its actor associations.");
  }

  observe(name: string, args: Record<string, unknown>, result: Record<string, unknown>, epoch: number): void {
    if (epoch !== this.epoch) return;
    if (name === "investigate.case.open") {
      const value = result.case as Record<string, unknown> | undefined;
      if (!value || typeof value.case_id !== "string" || !Number.isSafeInteger(value.revision) ||
          Number(value.revision) < 1 || value.case_id !== args.case_id) return;
      this.caseId = value.case_id; this.revision = Number(value.revision);
    } else if (result.case_id === this.caseId && Number.isSafeInteger(result.revision) && Number(result.revision) > this.revision) {
      this.revision = Number(result.revision);
    }
  }
}
