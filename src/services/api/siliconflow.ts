/**
 * SiliconFlow API adapter.
 *
 * Creates a fake `Anthropic` client object that translates Anthropic SDK calls
 * (beta.messages.create / stream) into OpenAI-compatible chat completions
 * requests and streams the responses back as Anthropic BetaRawMessageStreamEvent.
 *
 * This allows the rest of the codebase (claude.ts, query.ts, etc.) to treat
 * SiliconFlow models identically to Anthropic models.
 */
import type Anthropic from '@anthropic-ai/sdk'
import type {
  BetaContentBlockParam,
  BetaMessage,
  BetaMessageStreamParams,
  BetaRawMessageStreamEvent,
  BetaToolUnion,
  BetaMessageParam as MessageParam,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type { Stream } from '@anthropic-ai/sdk/streaming.mjs'
import {
  getSiliconFlowApiKey,
  getSiliconFlowBaseUrl,
  getSiliconFlowModelId,
} from 'src/utils/model/siliconflow.js'

// ---------------------------------------------------------------------------
// OpenAI-compatible types (subset needed for translation)
// ---------------------------------------------------------------------------

interface OAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | OAIContentPart[] | null
  tool_calls?: OAIToolCall[]
  tool_call_id?: string
  name?: string
}

interface OAIContentPart {
  type: 'text' | 'image_url'
  text?: string
  image_url?: { url: string }
}

interface OAIToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface OAITool {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters?: Record<string, unknown>
  }
}

interface OAIChatCompletionRequest {
  model: string
  messages: OAIMessage[]
  tools?: OAITool[]
  tool_choice?:
    | 'auto'
    | 'none'
    | { type: 'function'; function: { name: string } }
  max_tokens?: number
  temperature?: number
  stream?: boolean
  stream_options?: { include_usage: boolean }
}

interface OAIStreamChunk {
  id: string
  object: string
  created: number
  model: string
  choices: {
    index: number
    delta: {
      role?: string
      content?: string | null
      reasoning_content?: string | null
      tool_calls?: {
        index: number
        id?: string
        type?: string
        function?: { name?: string; arguments?: string }
      }[]
    }
    finish_reason: string | null
  }[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

// ---------------------------------------------------------------------------
// Request translation: Anthropic → OpenAI
// ---------------------------------------------------------------------------

function translateSystemPrompt(
  system: BetaMessageStreamParams['system'],
): string {
  if (!system) return ''
  if (typeof system === 'string') return system
  // Array of text/cache blocks – concatenate text parts
  return (system as Array<{ type: string; text?: string }>)
    .filter(b => b.type === 'text' && b.text)
    .map(b => b.text!)
    .join('\n\n')
}

function translateMessages(messages: MessageParam[]): OAIMessage[] {
  const result: OAIMessage[] = []

  for (const msg of messages) {
    if (msg.role === 'user') {
      result.push(...translateUserMessage(msg))
    } else if (msg.role === 'assistant') {
      result.push(translateAssistantMessage(msg))
    }
  }
  return result
}

function translateUserMessage(msg: MessageParam): OAIMessage[] {
  const out: OAIMessage[] = []
  if (typeof msg.content === 'string') {
    out.push({ role: 'user', content: msg.content })
    return out
  }

  // Complex content blocks – split tool_result into separate "tool" role messages
  const textParts: string[] = []
  for (const block of msg.content as BetaContentBlockParam[]) {
    switch ((block as any).type) {
      case 'text':
        textParts.push((block as any).text)
        break
      case 'tool_result': {
        // Flush accumulated text first
        if (textParts.length > 0) {
          out.push({ role: 'user', content: textParts.join('\n') })
          textParts.length = 0
        }
        const tr = block as any
        let content = ''
        if (typeof tr.content === 'string') {
          content = tr.content
        } else if (Array.isArray(tr.content)) {
          content = tr.content
            .filter((c: any) => c.type === 'text')
            .map((c: any) => c.text)
            .join('\n')
        }
        if (tr.is_error) {
          content = `[Tool Error] ${content}`
        }
        out.push({
          role: 'tool',
          tool_call_id: tr.tool_use_id,
          content,
        })
        break
      }
      case 'image': {
        const img = block as any
        if (img.source?.type === 'base64') {
          textParts.push(`[Image: ${img.source.media_type}]`)
        }
        break
      }
      default:
        // Skip unknown block types (document, tool_reference, etc.)
        break
    }
  }
  if (textParts.length > 0) {
    out.push({ role: 'user', content: textParts.join('\n') })
  }
  return out
}

function translateAssistantMessage(msg: MessageParam): OAIMessage {
  if (typeof msg.content === 'string') {
    return { role: 'assistant', content: msg.content }
  }

  const textParts: string[] = []
  const toolCalls: OAIToolCall[] = []
  for (const block of msg.content as any[]) {
    switch (block.type) {
      case 'text':
        textParts.push(block.text)
        break
      case 'thinking':
        // Include thinking as text for context
        if (block.thinking) {
          textParts.push(`<thinking>${block.thinking}</thinking>`)
        }
        break
      case 'tool_use':
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments:
              typeof block.input === 'string'
                ? block.input
                : JSON.stringify(block.input),
          },
        })
        break
      default:
        break
    }
  }

  const result: OAIMessage = {
    role: 'assistant',
    content: textParts.join('\n') || null,
  }
  if (toolCalls.length > 0) {
    result.tool_calls = toolCalls
  }
  return result
}

function translateTools(tools: BetaToolUnion[]): OAITool[] {
  const result: OAITool[] = []
  for (const tool of tools) {
    // Skip non-standard tool types (server_tool_use, advisor, etc.)
    if ((tool as any).type && (tool as any).type !== 'custom') continue
    const t = tool as any
    if (!t.name) continue
    // Skip deferred tools
    if (t.defer_loading) continue
    result.push({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      },
    })
  }
  return result
}

function translateToolChoice(
  choice: BetaMessageStreamParams['tool_choice'],
): OAIChatCompletionRequest['tool_choice'] {
  if (!choice) return undefined
  const c = choice as any
  if (c.type === 'auto') return 'auto'
  if (c.type === 'none') return 'none'
  if (c.type === 'tool' && c.name) {
    return { type: 'function', function: { name: c.name } }
  }
  return 'auto'
}

function buildOAIRequest(
  params: BetaMessageStreamParams & { stream?: boolean },
): OAIChatCompletionRequest {
  const systemText = translateSystemPrompt(params.system)
  const messages = translateMessages(params.messages)
  if (systemText) {
    messages.unshift({ role: 'system', content: systemText })
  }

  const tools = params.tools ? translateTools(params.tools) : undefined
  const toolChoice =
    tools && tools.length > 0
      ? translateToolChoice(params.tool_choice)
      : undefined

  return {
    model: getSiliconFlowModelId(params.model),
    messages,
    ...(tools && tools.length > 0 && { tools }),
    ...(toolChoice && { tool_choice: toolChoice }),
    max_tokens: params.max_tokens,
    ...(params.temperature !== undefined && {
      temperature: params.temperature,
    }),
    stream: params.stream ?? false,
    ...(params.stream && { stream_options: { include_usage: true } }),
  }
}

// ---------------------------------------------------------------------------
// Response translation: OpenAI → Anthropic (streaming)
// ---------------------------------------------------------------------------

/**
 * Creates an async iterable that reads an OpenAI-compatible SSE stream and
 * yields Anthropic BetaRawMessageStreamEvent events.
 */
async function* translateStream(
  response: Response,
  model: string,
): AsyncGenerator<BetaRawMessageStreamEvent> {
  const messageId = `msg_sf_${Date.now()}`
  let contentBlockIndex = 0
  let hasStartedTextBlock = false
  let hasStartedThinkingBlock = false
  let accumulatedToolCalls = new Map<
    number,
    { id: string; name: string; arguments: string }
  >()
  let toolCallBlockIndices = new Map<number, number>() // oai index → anthropic block index
  let inputTokens = 0
  let outputTokens = 0

  // Emit message_start
  yield {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  } as any

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()! // keep incomplete last line

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith(':')) continue
        if (trimmed === 'data: [DONE]') continue
        if (!trimmed.startsWith('data: ')) continue

        let chunk: OAIStreamChunk
        try {
          chunk = JSON.parse(trimmed.slice(6))
        } catch {
          continue
        }

        // Handle usage info (may come in the last chunk)
        if (chunk.usage) {
          inputTokens = chunk.usage.prompt_tokens
          outputTokens = chunk.usage.completion_tokens
        }

        const choice = chunk.choices?.[0]
        if (!choice) continue
        const delta = choice.delta

        // Handle reasoning_content (e.g. DeepSeek-R1)
        if (delta.reasoning_content) {
          if (!hasStartedThinkingBlock) {
            hasStartedThinkingBlock = true
            yield {
              type: 'content_block_start',
              index: contentBlockIndex,
              content_block: {
                type: 'thinking',
                thinking: '',
                signature: '',
              },
            } as any
          }
          yield {
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: {
              type: 'thinking_delta',
              thinking: delta.reasoning_content,
            },
          } as any
        }

        // Handle text content
        if (delta.content) {
          // If we had a thinking block and now getting content, close thinking first
          if (hasStartedThinkingBlock && !hasStartedTextBlock) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            } as any
            contentBlockIndex++
            hasStartedThinkingBlock = false // closed
          }

          if (!hasStartedTextBlock) {
            hasStartedTextBlock = true
            yield {
              type: 'content_block_start',
              index: contentBlockIndex,
              content_block: { type: 'text', text: '' },
            } as any
          }
          yield {
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: { type: 'text_delta', text: delta.content },
          } as any
        }

        // Handle tool calls
        if (delta.tool_calls) {
          // Close text block before tool calls if open
          if (hasStartedTextBlock) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            } as any
            contentBlockIndex++
            hasStartedTextBlock = false
          }
          // Close thinking block if still open
          if (hasStartedThinkingBlock) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            } as any
            contentBlockIndex++
            hasStartedThinkingBlock = false
          }

          for (const tc of delta.tool_calls) {
            const tcIdx = tc.index
            if (!accumulatedToolCalls.has(tcIdx)) {
              // New tool call
              accumulatedToolCalls.set(tcIdx, {
                id: tc.id || `toolu_sf_${Date.now()}_${tcIdx}`,
                name: tc.function?.name || '',
                arguments: tc.function?.arguments || '',
              })
              toolCallBlockIndices.set(tcIdx, contentBlockIndex)
              yield {
                type: 'content_block_start',
                index: contentBlockIndex,
                content_block: {
                  type: 'tool_use',
                  id: accumulatedToolCalls.get(tcIdx)!.id,
                  name: tc.function?.name || '',
                  input: {},
                },
              } as any
              if (tc.function?.arguments) {
                yield {
                  type: 'content_block_delta',
                  index: contentBlockIndex,
                  delta: {
                    type: 'input_json_delta',
                    partial_json: tc.function.arguments,
                  },
                } as any
              }
              contentBlockIndex++
            } else {
              // Continuation of existing tool call
              const existing = accumulatedToolCalls.get(tcIdx)!
              const blockIdx = toolCallBlockIndices.get(tcIdx)!
              if (tc.function?.name) {
                existing.name += tc.function.name
              }
              if (tc.function?.arguments) {
                existing.arguments += tc.function.arguments
                yield {
                  type: 'content_block_delta',
                  index: blockIdx,
                  delta: {
                    type: 'input_json_delta',
                    partial_json: tc.function.arguments,
                  },
                } as any
              }
            }
          }
        }

        // Handle finish
        if (choice.finish_reason) {
          // Close any open blocks
          if (hasStartedThinkingBlock) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            } as any
            contentBlockIndex++
          }
          if (hasStartedTextBlock) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            } as any
            contentBlockIndex++
          }

          // Close any open tool call blocks
          for (const [tcIdx] of accumulatedToolCalls) {
            const blockIdx = toolCallBlockIndices.get(tcIdx)!
            yield {
              type: 'content_block_stop',
              index: blockIdx,
            } as any
          }

          // Map stop reason
          let stopReason: string
          switch (choice.finish_reason) {
            case 'stop':
              stopReason = 'end_turn'
              break
            case 'tool_calls':
              stopReason = 'tool_use'
              break
            case 'length':
              stopReason = 'max_tokens'
              break
            default:
              stopReason = 'end_turn'
          }

          yield {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: {
              input_tokens: inputTokens,
              output_tokens: outputTokens,
            },
          } as any

          yield { type: 'message_stop' } as any
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * Translate a non-streaming OpenAI response to an Anthropic BetaMessage.
 */
function translateNonStreamingResponse(data: any, model: string): BetaMessage {
  const choice = data.choices?.[0]
  const message = choice?.message
  const content: any[] = []

  if (message?.content) {
    content.push({ type: 'text', text: message.content })
  }

  if (message?.tool_calls) {
    for (const tc of message.tool_calls) {
      let input: Record<string, unknown> = {}
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {
        input = { _raw: tc.function.arguments }
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      })
    }
  }

  let stopReason: string
  switch (choice?.finish_reason) {
    case 'stop':
      stopReason = 'end_turn'
      break
    case 'tool_calls':
      stopReason = 'tool_use'
      break
    case 'length':
      stopReason = 'max_tokens'
      break
    default:
      stopReason = 'end_turn'
  }

  return {
    id: data.id || `msg_sf_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: data.usage?.prompt_tokens ?? 0,
      output_tokens: data.usage?.completion_tokens ?? 0,
    },
  } as any
}

// ---------------------------------------------------------------------------
// SiliconFlow fetch helper
// ---------------------------------------------------------------------------

async function siliconFlowFetch(
  path: string,
  body: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<Response> {
  const baseUrl = getSiliconFlowBaseUrl()
  const apiKey = getSiliconFlowApiKey()
  const url = `${baseUrl}${path}`

  return fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal,
  })
}

// ---------------------------------------------------------------------------
// Fake Anthropic client for SiliconFlow
// ---------------------------------------------------------------------------

/**
 * Creates an object that duck-types as an Anthropic SDK client but routes
 * requests to SiliconFlow's OpenAI-compatible API.
 *
 * Only the methods actually used by claude.ts are implemented:
 * - `beta.messages.create(params)` (non-streaming)
 * - `beta.messages.create(params, opts).withResponse()` (streaming)
 */
export function createSiliconFlowClient(): Anthropic {
  const client = {
    beta: {
      messages: {
        create(
          params: BetaMessageStreamParams & { stream?: boolean },
          options?: {
            signal?: AbortSignal
            headers?: Record<string, string>
            timeout?: number
          },
        ): any {
          const isStreaming = params.stream === true

          if (isStreaming) {
            // Return a thenable with .withResponse()
            const promise = doStreamingRequest(params, options)
            // The SDK pattern is: create(...).withResponse()
            // withResponse() returns { data: Stream, response, request_id }
            const thenable: any = promise
            thenable.withResponse = () => promise
            return thenable
          } else {
            // Non-streaming: return a promise of BetaMessage
            return doNonStreamingRequest(params, options)
          }
        },
      },
    },
  }

  return client as unknown as Anthropic
}

async function doStreamingRequest(
  params: BetaMessageStreamParams & { stream?: boolean },
  options?: {
    signal?: AbortSignal
    headers?: Record<string, string>
    timeout?: number
  },
): Promise<{
  data: Stream<BetaRawMessageStreamEvent>
  response: Response
  request_id: string
}> {
  const oaiRequest = buildOAIRequest({ ...params, stream: true })
  const response = await siliconFlowFetch(
    '/chat/completions',
    oaiRequest as unknown as Record<string, unknown>,
    options?.signal,
  )

  if (!response.ok) {
    const errorBody = await response.text()
    throw new Error(`SiliconFlow API error (${response.status}): ${errorBody}`)
  }

  const model = getSiliconFlowModelId(params.model)

  // Create a Stream-like async iterable
  const stream = translateStream(response, model)
  const streamWrapper: any = {
    [Symbol.asyncIterator]() {
      return stream[Symbol.asyncIterator]()
    },
    // The SDK accesses .controller for type detection in claude.ts
    controller: new AbortController(),
    response,
  }

  return {
    data: streamWrapper as Stream<BetaRawMessageStreamEvent>,
    response,
    request_id: response.headers.get('x-request-id') || `sf_${Date.now()}`,
  }
}

async function doNonStreamingRequest(
  params: BetaMessageStreamParams & { stream?: boolean },
  options?: { signal?: AbortSignal; timeout?: number },
): Promise<BetaMessage> {
  const oaiRequest = buildOAIRequest({ ...params, stream: false })
  const response = await siliconFlowFetch(
    '/chat/completions',
    oaiRequest as unknown as Record<string, unknown>,
    options?.signal,
  )

  if (!response.ok) {
    const errorBody = await response.text()
    throw new Error(`SiliconFlow API error (${response.status}): ${errorBody}`)
  }

  const data = await response.json()
  return translateNonStreamingResponse(
    data,
    getSiliconFlowModelId(params.model),
  )
}
