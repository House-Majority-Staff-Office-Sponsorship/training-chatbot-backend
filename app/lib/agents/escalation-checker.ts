/**
 * EscalationChecker — custom BaseAgent for loop control.
 *
 * Checks the research evaluation state. If the evaluator graded "pass",
 * emits an escalation event to break out of the LoopAgent. Otherwise
 * yields an empty event to let the loop continue.
 */

import { BaseAgent, createEvent, createEventActions } from "@google/adk";
import type { InvocationContext, Event } from "@google/adk";

export class EscalationChecker extends BaseAgent {
  constructor() {
    super({
      name: "escalation_checker",
      description:
        "Checks if research evaluation passed and escalates to stop the refinement loop.",
    });
  }

  protected async *runAsyncImpl(
    ctx: InvocationContext
  ): AsyncGenerator<Event, void, void> {
    const evaluation = ctx.session.state?.["research_evaluation"];
    let grade: string | undefined;

    if (typeof evaluation === "string") {
      try {
        grade = JSON.parse(evaluation).grade;
      } catch {
        grade = undefined;
      }
    } else if (evaluation && typeof evaluation === "object") {
      grade = (evaluation as Record<string, unknown>).grade as string;
    }

    if (grade === "pass") {
      yield createEvent({
        author: this.name,
        actions: { ...createEventActions(), escalate: true },
      });
    } else {
      yield createEvent({ author: this.name });
    }
  }

  protected async *runLiveImpl(
    _ctx: InvocationContext
  ): AsyncGenerator<Event, void, void> {
    // Live mode not used — no-op
  }
}
