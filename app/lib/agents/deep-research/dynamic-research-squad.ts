/**
 * DynamicResearchSquad — custom BaseAgent that spawns one researcher
 * per research question and runs them all in parallel.
 *
 * Reads `research_questions` (JSON) from session state, parses questions,
 * creates an LlmAgent + RAG tool for each, runs them concurrently, and
 * emits individual completion events as each finishes (for live frontend display)
 * before writing combined findings to state for the report composer.
 */

import {
  BaseAgent,
  LlmAgent,
  InMemoryRunner,
  createEvent,
  createEventActions,
} from "@google/adk";
import type { InvocationContext, Event } from "@google/adk";
import { createRagRetrievalTool, type RagTokenUsage } from "../shared-tools/rag-tool";

export interface DynamicResearchSquadConfig {
  project: string;
  location: string;
  model: string;
  ragCorpus: string;
}

export class DynamicResearchSquad extends BaseAgent {
  private config: DynamicResearchSquadConfig;

  constructor(config: DynamicResearchSquadConfig) {
    super({
      name: "dynamic_research_squad",
      description:
        "Dynamically spawns one researcher per research question and runs them all in parallel.",
    });
    this.config = config;
  }

  /**
   * Extract questions from the question expander output.
   * Tries JSON first, then falls back to extracting question-like lines
   * from any text format (numbered lists, headings, Roman numerals, etc.).
   * Hard cap at 10 questions.
   */
  private parseQuestions(
    raw: string
  ): { question: string; description: string }[] {
    // 1. Try JSON extraction
    const jsonMatch = raw.match(
      /\{[\s\S]*"questions"\s*:\s*\[[\s\S]*\]\s*\}/
    );
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0]);
        if (Array.isArray(parsed?.questions) && parsed.questions.length > 0) {
          return parsed.questions
            .filter(
              (q: { question?: string }) =>
                typeof q.question === "string" && q.question.trim()
            )
            .slice(0, 10)
            .map((q: { question: string; description?: string }) => ({
              question: q.question.trim(),
              description:
                typeof q.description === "string" ? q.description.trim() : "",
            }));
        }
      } catch {
        // JSON parse failed — try text fallback
      }
    }

    // 2. Fallback: extract question-like lines from any text format.
    //    Matches lines that end with "?" and strips leading bullets/numbers/headings.
    const questions: { question: string; description: string }[] = [];
    for (const line of raw.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.includes("?")) continue;

      // Strip leading markers: *, -, #, numbers, letters, Roman numerals, etc.
      const cleaned = trimmed
        .replace(
          /^(?:[-*•]\s*|#{1,4}\s+|\d+[.)]\s*|[a-zA-Z][.)]\s*|\*{1,2}|[IVXLC]+[.)]\s*)+/,
          ""
        )
        .trim();

      if (cleaned.length > 10 && cleaned.includes("?")) {
        questions.push({ question: cleaned, description: "" });
      }
    }

    return questions.slice(0, 10);
  }

  /**
   * Run a single question researcher and return its findings + logs.
   */
  private async runQuestionResearcher(
    item: { question: string; description: string },
    enrichedQuery: string
  ): Promise<{ findings: string; logs: Array<{ agent: string; message: string; promptTokens: number; responseTokens: number; totalTokens: number; timestamp: number }> }> {
    type ResearcherLog = { agent: string; message: string; promptTokens: number; responseTokens: number; totalTokens: number; timestamp: number };
    const ragLogs: ResearcherLog[] = [];
    const ragTool = createRagRetrievalTool({
      project: this.config.project,
      location: this.config.location,
      model: this.config.model,
      ragCorpus: this.config.ragCorpus,
      onTokenUsage: (usage: RagTokenUsage) => {
        ragLogs.push({
          agent: "retrieve_from_rag",
          message: `retrieve_from_rag → "${usage.query.length > 60 ? usage.query.slice(0, 57) + "..." : usage.query}"`,
          promptTokens: usage.promptTokens,
          responseTokens: usage.responseTokens,
          totalTokens: usage.totalTokens,
          timestamp: usage.timestamp,
        });
      },
    });

    const safeName = `researcher_${item.question
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .slice(0, 50)}`;

    const agent = new LlmAgent({
      name: safeName,
      model: this.config.model,
      description: `Researches: "${item.question}"`,
      instruction: `You are a focused research agent for the House Majority Staff Office training documentation system. Your job is to thoroughly research ONE specific question.

QUESTION TO RESEARCH:
${item.question}

RESEARCH GUIDANCE:
${item.description}

OVERALL CONTEXT:
${enrichedQuery}

Instructions:
1. Formulate 3-5 targeted queries specifically about this question.
2. Call the retrieve_from_rag tool for EACH query.
3. Synthesize ALL results into a comprehensive, detailed answer.

CRITICAL — Evidence and source requirements:
- Always include specific policy numbers, section references, or document titles when found.
- Quote key definitions, rules, and requirements verbatim using quotation marks.
- Cite data points, dates, thresholds, and numerical values exactly as they appear in the source.
- If a finding lacks a specific reference, note that no source reference was found.
- If the RAG corpus has no relevant information for this question, clearly state that.
- The RAG tool returns a SOURCES section with each response. You MUST collect ALL source references from every RAG call.

FORMATTING:
- Write in plain text only. No markdown, no headers, no bold, no bullet points, no special formatting.
- Use simple numbered lists or line breaks to organize information.
- The final report composer will handle all formatting later.

Your response MUST end with a REFERENCES section listing ALL sources from the RAG responses, one per line:

REFERENCES:
Document/source title (URI if available)
Document/source title (URI if available)`,
      tools: [ragTool],
      outputKey: "question_findings",
    });

    const runner = new InMemoryRunner({ agent });
    let findings = "";
    const logs: Array<{ agent: string; message: string; promptTokens: number; responseTokens: number; totalTokens: number; timestamp: number }> = [];

    for await (const event of runner.runEphemeral({
      userId: "question-researcher",
      newMessage: {
        role: "user",
        parts: [{ text: item.question }],
      },
    })) {
      const author = event.author ?? "unknown";
      if (author !== "user") {
        const usage = event.usageMetadata;
        const promptTokens = usage?.promptTokenCount ?? 0;
        const responseTokens = usage?.candidatesTokenCount ?? 0;
        const totalTokens = usage?.totalTokenCount ?? (promptTokens + responseTokens);
        const stateKeys = event.actions?.stateDelta ? Object.keys(event.actions.stateDelta) : [];
        const detail = stateKeys.length > 0 ? ` → wrote [${stateKeys.join(", ")}]` : "";
        // Skip noise events (e.g. "tool response") that have no tokens and no state writes
        if (totalTokens > 0 || stateKeys.length > 0) {
          logs.push({
            agent: author,
            message: `${author}${detail}`,
            promptTokens,
            responseTokens,
            totalTokens,
            timestamp: Date.now(),
          });
        }
      }

      const delta = event.actions?.stateDelta;
      if (delta && typeof delta["question_findings"] === "string") {
        findings = delta["question_findings"];
      }
    }

    // Merge RAG tool token logs into the timeline
    const allLogs = [...logs, ...ragLogs].sort((a, b) => a.timestamp - b.timestamp);
    return { findings, logs: allLogs };
  }

  protected async *runAsyncImpl(
    ctx: InvocationContext
  ): AsyncGenerator<Event, void, void> {
    const researchQuestions =
      (ctx.session.state?.["research_questions"] as string) ?? "";
    const enrichedQuery =
      (ctx.session.state?.["enriched_query"] as string) ?? "";

    const questions = this.parseQuestions(researchQuestions);

    // Fallback: if parsing fails, run a single generic researcher
    if (questions.length === 0) {
      const result = await this.runQuestionResearcher(
        { question: "General Research", description: researchQuestions },
        enrichedQuery
      );

      // Emit researcher logs
      for (const log of result.logs) {
        yield createEvent({
          author: this.name,
          actions: {
            ...createEventActions(),
            stateDelta: { [`researcher_log`]: JSON.stringify({ ...log, researcherIndex: 0 }) },
          },
        });
      }

      if (ctx.session.state) {
        ctx.session.state["section_findings_all"] = result.findings;
      }

      yield createEvent({
        author: this.name,
        content: { parts: [{ text: result.findings }] },
        actions: {
          ...createEventActions(),
          stateDelta: {
            researcher_count: "1",
            researcher_labels: JSON.stringify(["General Research"]),
            researcher_0: result.findings,
          },
        },
      });
      return;
    }

    // Emit researcher count + labels so the frontend can create slots
    yield createEvent({
      author: this.name,
      actions: {
        ...createEventActions(),
        stateDelta: {
          researcher_count: String(questions.length),
          researcher_labels: JSON.stringify(
            questions.map((q) => q.question)
          ),
        },
      },
    });

    // ── Completion queue for yielding events as researchers finish ──
    type ResearcherLog = { agent: string; message: string; promptTokens: number; responseTokens: number; totalTokens: number; timestamp: number };
    type CompletionItem = { index: number; findings: string; logs: ResearcherLog[] };
    const completedBuffer: CompletionItem[] = [];
    const waiters: Array<{ resolve: (item: CompletionItem) => void }> = [];

    function pushCompletion(item: CompletionItem) {
      if (waiters.length > 0) {
        waiters.shift()!.resolve(item);
      } else {
        completedBuffer.push(item);
      }
    }

    function nextCompletion(): Promise<CompletionItem> {
      if (completedBuffer.length > 0) {
        return Promise.resolve(completedBuffer.shift()!);
      }
      return new Promise((resolve) => waiters.push({ resolve }));
    }

    // Launch all researchers in parallel
    const results: string[] = new Array(questions.length).fill("");

    for (let i = 0; i < questions.length; i++) {
      this.runQuestionResearcher(questions[i], enrichedQuery)
        .then((result) => pushCompletion({ index: i, findings: result.findings, logs: result.logs }))
        .catch((err) =>
          pushCompletion({
            index: i,
            findings: `[Research failed: ${err instanceof Error ? err.message : String(err)}]`,
            logs: [],
          })
        );
    }

    // Yield events as each researcher completes (in completion order)
    for (let done = 0; done < questions.length; done++) {
      const { index, findings, logs } = await nextCompletion();
      results[index] = findings;

      // Emit researcher logs so the outer runner can forward them
      for (const log of logs) {
        yield createEvent({
          author: this.name,
          actions: {
            ...createEventActions(),
            stateDelta: { [`researcher_log`]: JSON.stringify({ ...log, researcherIndex: index }) },
          },
        });
      }

      yield createEvent({
        author: this.name,
        actions: {
          ...createEventActions(),
          stateDelta: {
            [`researcher_${index}`]: findings,
          },
        },
      });
    }

    // Combine all findings and write to session state for report composer
    const combinedFindings = questions
      .map((q, i) => `${q.question}\n\n${results[i]}`)
      .join("\n\n---\n\n");

    if (ctx.session.state) {
      ctx.session.state["section_findings_all"] = combinedFindings;
    }
  }

  protected async *runLiveImpl(
    _ctx: InvocationContext
  ): AsyncGenerator<Event, void, void> {
    // Live mode not used — no-op
  }
}
