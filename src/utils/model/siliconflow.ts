/**
 * SiliconFlow provider configuration and model detection.
 *
 * SiliconFlow uses an OpenAI-compatible API (POST /v1/chat/completions).
 * This module provides helpers for detecting SiliconFlow models, reading
 * configuration from environment variables, and listing available models.
 */

/** Well-known SiliconFlow model IDs shown in the /model picker. */
export const SILICONFLOW_KNOWN_MODELS: readonly string[] = [
  'Pro/zai-org/GLM-5',
  'deepseek-ai/DeepSeek-V3',
  'deepseek-ai/DeepSeek-R1',
  'Qwen/Qwen3-235B-A22B',
  'Qwen/Qwen2.5-72B-Instruct',
] as const

/**
 * Prefix used to tag SiliconFlow models in the app state so they can be
 * distinguished from Anthropic models at runtime without maintaining an
 * exhaustive list.  A model value stored as `siliconflow:deepseek-ai/DeepSeek-V3`
 * will be detected by `isSiliconFlowModel()` and the prefix stripped before
 * sending to the SiliconFlow API.
 */
export const SILICONFLOW_MODEL_PREFIX = 'siliconflow:'

/**
 * Returns true when the given model string represents a SiliconFlow model.
 * Detection is based on the `siliconflow:` prefix that we attach when the
 * user selects a SiliconFlow model from the picker.
 */
export function isSiliconFlowModel(model: string | null | undefined): boolean {
  if (!model) return false
  return model.startsWith(SILICONFLOW_MODEL_PREFIX)
}

/**
 * Strips the `siliconflow:` prefix and returns the raw model ID that
 * SiliconFlow's API expects (e.g. `deepseek-ai/DeepSeek-V3`).
 */
export function getSiliconFlowModelId(model: string): string {
  if (model.startsWith(SILICONFLOW_MODEL_PREFIX)) {
    return model.slice(SILICONFLOW_MODEL_PREFIX.length)
  }
  return model
}

/**
 * Returns the SiliconFlow API key from environment, or undefined if not
 * configured.
 */
export function getSiliconFlowApiKey(): string | undefined {
  return process.env.SILICONFLOW_API_KEY
}

/**
 * Returns the SiliconFlow API base URL.
 * Defaults to `https://api.siliconflow.cn/v1`.
 */
export function getSiliconFlowBaseUrl(): string {
  return process.env.SILICONFLOW_BASE_URL || 'https://api.siliconflow.cn/v1'
}

/**
 * Whether SiliconFlow is available (API key is configured).
 */
export function isSiliconFlowAvailable(): boolean {
  return !!getSiliconFlowApiKey()
}

/**
 * Returns the default SiliconFlow model from env, or the first known model.
 */
export function getDefaultSiliconFlowModel(): string {
  return process.env.SILICONFLOW_MODEL || SILICONFLOW_KNOWN_MODELS[0]!
}

/**
 * Returns a human-friendly display name for a SiliconFlow model ID.
 */
export function getSiliconFlowModelDisplayName(modelId: string): string {
  // Strip the prefix if present
  const raw = getSiliconFlowModelId(modelId)
  // Take the part after the last `/` as the short name
  const parts = raw.split('/')
  return parts[parts.length - 1] || raw
}
