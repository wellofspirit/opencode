import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { Schema } from "effect"
import { HttpApi, HttpApiEndpoint, HttpApiGroup, OpenApi } from "effect/unstable/httpapi"
import { ModelNotFoundError, UpstreamError } from "../errors"
import { Authorization } from "../middleware/authorization"
import { InstanceContextMiddleware } from "../middleware/instance-context"
import { WorkspaceRoutingMiddleware, WorkspaceRoutingQuery } from "../middleware/workspace-routing"
import { described } from "./metadata"

export const JudgePaths = {
  completion: "/judge/completion",
} as const

export const JudgeCompletionPayload = Schema.Struct({
  model: Schema.Struct({
    providerID: ProviderV2.ID,
    modelID: ModelV2.ID,
  }),
  system: Schema.String,
  user: Schema.String,
  maxTokens: Schema.optional(Schema.Number),
  stopSequences: Schema.optional(Schema.Array(Schema.String)),
}).annotate({ identifier: "JudgeCompletionInput" })

const JudgeCompletionResponse = Schema.Struct({
  text: Schema.String,
}).annotate({ identifier: "JudgeCompletionOutput" })

export const JudgeApi = HttpApi.make("judge")
  .add(
    HttpApiGroup.make("judge")
      .add(
        HttpApiEndpoint.post("completion", JudgePaths.completion, {
          query: WorkspaceRoutingQuery,
          payload: JudgeCompletionPayload,
          success: described(JudgeCompletionResponse, "Generated text"),
          error: [ModelNotFoundError, UpstreamError],
        }).annotateMerge(
          OpenApi.annotations({
            identifier: "judge.completion",
            summary: "Tool-less single-turn completion",
            description:
              "Run one system+user completion straight against the resolved provider model. Creates no session, registers no tools and never consults the permission layer, so the caller cannot be steered into executing anything. maxTokens is clamped to the model's output ceiling and stopSequences are passed through, except where the provider rejects them (the ChatGPT/Codex OAuth backend and Copilot gpt-* reject an output cap; OpenAI's Responses API has no stop-sequence parameter).",
          }),
        ),
      )
      .annotateMerge(
        OpenApi.annotations({
          title: "judge",
          description: "Tool-less completion routes.",
        }),
      )
      .middleware(InstanceContextMiddleware)
      .middleware(WorkspaceRoutingMiddleware)
      .middleware(Authorization),
  )
  .annotateMerge(
    OpenApi.annotations({
      title: "opencode judge HttpApi",
      version: "0.0.1",
      description: "Tool-less completion surface for security-judge style callers.",
    }),
  )
