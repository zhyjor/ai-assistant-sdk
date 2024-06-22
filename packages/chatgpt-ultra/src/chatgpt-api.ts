import Keyv from 'keyv'
import pTimeout from 'p-timeout'
import QuickLRU from 'quick-lru'
import { v4 as uuidv4 } from 'uuid'
import OpenAI from 'openai'

import * as tokenizer from './tokenizer'
import * as types from './types'
import { fetch as globalFetch } from './fetch'
import { fetchSSE } from './fetch-sse'

const CHATGPT_MODEL = 'gpt-3.5-turbo'

const USER_LABEL_DEFAULT = 'User'
const ASSISTANT_LABEL_DEFAULT = 'ChatGPT'
export class ChatGPTAPI {
  protected _apiKey: string
  protected _apiBaseUrl: string
  protected _apiOrg?: string
  protected _debug: boolean

  protected _systemMessage: string
  protected _completionParams: Omit<
    types.openai.CreateChatCompletionRequest,
    'messages' | 'n'
  >
  protected _maxModelTokens: number
  protected _maxResponseTokens: number
  protected _fetch: types.FetchFn

  protected _getMessageById: types.GetMessageByIdFunction
  protected _upsertMessage: types.UpsertMessageFunction

  protected _messageStore: Keyv<types.ChatMessage>;
  protected _client: OpenAI;

  /**
   * Creates a new client wrapper around OpenAI's chat completion API, mimicing the official ChatGPT webapp's functionality as closely as possible.
   *
   * @param apiKey - OpenAI API key (required).
   * @param apiOrg - Optional OpenAI API organization (optional).
   * @param baseURL - Optional override for the OpenAI API base URL.
   * @param debug - Optional enables logging debugging info to stdout.
   * @param completionParams - Param overrides to send to the [OpenAI chat completion API](https://platform.openai.com/docs/api-reference/chat/create). Options like `temperature` and `presence_penalty` can be tweaked to change the personality of the assistant.
   * @param maxModelTokens - Optional override for the maximum number of tokens allowed by the model's context. Defaults to 4096.
   * @param maxResponseTokens - Optional override for the minimum number of tokens allowed for the model's response. Defaults to 1000.
   * @param messageStore - Optional [Keyv](https://github.com/jaredwray/keyv) store to persist chat messages to. If not provided, messages will be lost when the process exits.
   * @param getMessageById - Optional function to retrieve a message by its ID. If not provided, the default implementation will be used (using an in-memory `messageStore`).
   * @param upsertMessage - Optional function to insert or update a message. If not provided, the default implementation will be used (using an in-memory `messageStore`).
   * @param fetch - Optional override for the `fetch` implementation to use. Defaults to the global `fetch` function.
   */
  constructor(opts: types.ChatGPTAPIOptions) {
    const {
      apiKey,
      apiOrg,
      baseURL = 'https://api.openai.com/v1',
      debug = false,
      messageStore,
      completionParams,
      systemMessage,
      maxModelTokens = 4000,
      maxResponseTokens = 1000,
      getMessageById,
      upsertMessage,
      fetch = globalFetch
    } = opts

    this._apiKey = apiKey
    this._apiOrg = apiOrg
    this._apiBaseUrl = baseURL
    this._debug = !!debug
    this._fetch = fetch

    this._completionParams = {
      model: CHATGPT_MODEL,
      temperature: 0.8,
      top_p: 1.0,
      presence_penalty: 1.0,
      ...completionParams
    }

    this._systemMessage = systemMessage

    if (this._systemMessage === undefined) {
      const currentDate = new Date().toISOString().split('T')[0]
      this._systemMessage = `You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: ${currentDate}`
    }

    this._maxModelTokens = maxModelTokens
    this._maxResponseTokens = maxResponseTokens

    this._getMessageById = getMessageById ?? this._defaultGetMessageById
    this._upsertMessage = upsertMessage ?? this._defaultUpsertMessage

    if (messageStore) {
      this._messageStore = messageStore
    } else {
      this._messageStore = new Keyv<types.ChatMessage, any>({
        store: new QuickLRU<string, types.ChatMessage>({ maxSize: 10000 })
      })
    }

    if (!this._apiKey) {
      throw new Error('OpenAI missing required apiKey')
    }

    if (!this._fetch) {
      throw new Error('Invalid environment; fetch is not defined')
    }

    if (typeof this._fetch !== 'function') {
      throw new Error('Invalid "fetch" is not a function')
    }
    // 初始化client
    this._client = new OpenAI({
      baseURL,
      apiKey, // This is the default and can be omitted
    })
  }

  /**
   * Sends a message to the OpenAI chat completions endpoint, waits for the response
   * to resolve, and returns the response.
   *
   * If you want your response to have historical context, you must provide a valid `parentMessageId`.
   *
   * If you want to receive a stream of partial responses, use `opts.onProgress`.
   *
   * Set `debug: true` in the `ChatGPTAPI` constructor to log more info on the full prompt sent to the OpenAI chat completions API. You can override the `systemMessage` in `opts` to customize the assistant's instructions.
   *
   * @param message - The prompt message to send
   * @param opts.parentMessageId - Optional ID of the previous message in the conversation (defaults to `undefined`)
   * @param opts.conversationId - Optional ID of the conversation (defaults to `undefined`)
   * @param opts.messageId - Optional ID of the message to send (defaults to a random UUID)
   * @param opts.systemMessage - Optional override for the chat "system message" which acts as instructions to the model (defaults to the ChatGPT system message)
   * @param opts.timeoutMs - Optional timeout in milliseconds (defaults to no timeout)
   * @param opts.onProgress - Optional callback which will be invoked every time the partial response is updated
   * @param opts.abortSignal - Optional callback used to abort the underlying `fetch` call using an [AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController)
   * @param completionParams - Optional overrides to send to the [OpenAI chat completion API](https://platform.openai.com/docs/api-reference/chat/create). Options like `temperature` and `presence_penalty` can be tweaked to change the personality of the assistant.
   *
   * @returns The response from ChatGPT
   */
  async sendMessage(
    text: string,
    opts: types.SendMessageOptions = {}
  ): Promise<types.ChatMessage> {
    const {
      parentMessageId,
      messageId = uuidv4(),
      timeoutMs,
      onProgress,
      stream = onProgress ? true : false,
      completionParams,
      conversationId
    } = opts

    let { abortSignal } = opts

    let abortController: AbortController = null
    if (timeoutMs && !abortSignal) {
      abortController = new AbortController()
      abortSignal = abortController.signal
    }

    const message: types.ChatMessage = {
      role: 'user',
      id: messageId,
      conversationId,
      parentMessageId,
      text
    }

    const latestQuestion = message

    const { messages, maxTokens, numTokens } = await this._buildMessages(
      text,
      opts
    )

    const result: types.ChatMessage = {
      role: 'assistant',
      id: uuidv4(),
      conversationId,
      parentMessageId: messageId,
      text: ''
    }

    const responseP = new Promise<types.ChatMessage>(
      async (resolve, reject) => {
        const body = {
          max_tokens: maxTokens,
          ...this._completionParams,
          ...completionParams,
          messages,
          stream
        }

        if (this._debug) {
          console.log(`sendMessage (${numTokens} tokens)`, body)
        }

        if (stream) {
          try {
            // 处理stream场景
            const stream = await this._client.chat.completions.create({
              model: this._completionParams.model,
              messages: body.messages,
              max_tokens: body.max_tokens,
              stream: true,
            });
            for await (const chunk of stream) {
              // 判断是否结束
              const choiceItem = chunk?.choices?.[0];
              if (choiceItem?.finish_reason === 'stop') {
                result.text += choiceItem.delta.content;
                resolve(result);
              } else {
                result.delta = choiceItem.delta.content;
                result.text += choiceItem.delta.content;
                result.role = choiceItem.delta.role;
                onProgress?.(result);
              }
              console.log(chunk.choices[0]?.delta?.content);
            }
          } catch (e) {
            console.warn('OpenAI stream SEE event unexpected error', e)
            return reject(e)
          }

        } else {
          try {
            // 处理stream场景
            const res = await this._client.chat.completions.create({
              model: this._completionParams.model,
              messages: body.messages,
              max_tokens: body.max_tokens,
            });
            const choiceItem = res?.choices?.[0];
            if (choiceItem.finish_reason === 'stop') {
              result.id = res.id;
              result.text = choiceItem.message.content;
              result.role = choiceItem.message.role;
              result.detail = res;
              return resolve(result)
            } else {
              return reject(new Error(
                `OpenAI error ${choiceItem.finish_reason}`
              ))
            }
          } catch (err) {
            return reject(err)
          }
        }
      }
    ).then(async (message) => {
      if (message.detail && !message.detail.usage) {
        try {
          const promptTokens = numTokens
          const completionTokens = await this._getTokenCount(message.text)
          message.detail.usage = {
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens,
            total_tokens: promptTokens + completionTokens,
            estimated: true
          }
        } catch (err) {
          // TODO: this should really never happen, but if it does,
          // we should handle notify the user gracefully
        }
      }

      return Promise.all([
        this._upsertMessage(latestQuestion),
        this._upsertMessage(message)
      ]).then(() => message)
    })

    if (timeoutMs) {
      if (abortController) {
        // This will be called when a timeout occurs in order for us to forcibly
        // ensure that the underlying HTTP request is aborted.
        ; (responseP as any).cancel = () => {
          abortController.abort()
        }
      }

      return pTimeout(responseP, {
        milliseconds: timeoutMs,
        message: 'OpenAI timed out waiting for response'
      })
    } else {
      return responseP
    }
  }

  get apiKey(): string {
    return this._apiKey
  }

  set apiKey(apiKey: string) {
    this._apiKey = apiKey
  }

  get apiOrg(): string {
    return this._apiOrg
  }

  set apiOrg(apiOrg: string) {
    this._apiOrg = apiOrg
  }

  protected async _buildMessages(text: string, opts: types.SendMessageOptions) {
    const { systemMessage = this._systemMessage } = opts
    let { parentMessageId } = opts

    const userLabel = USER_LABEL_DEFAULT
    const assistantLabel = ASSISTANT_LABEL_DEFAULT

    const maxNumTokens = this._maxModelTokens - this._maxResponseTokens
    let messages: types.openai.ChatCompletionRequestMessage[] = []

    if (systemMessage) {
      messages.push({
        role: 'system',
        content: systemMessage
      })
    }

    const systemMessageOffset = messages.length
    let nextMessages = text
      ? messages.concat([
        {
          role: 'user',
          content: text,
          name: opts.name
        }
      ])
      : messages
    let numTokens = 0

    do {
      const prompt = nextMessages
        .reduce((prompt, message) => {
          switch (message.role) {
            case 'system':
              return prompt.concat([`Instructions:\n${message.content}`])
            case 'user':
              return prompt.concat([`${userLabel}:\n${message.content}`])
            default:
              return prompt.concat([`${assistantLabel}:\n${message.content}`])
          }
        }, [] as string[])
        .join('\n\n')

      const nextNumTokensEstimate = await this._getTokenCount(prompt)
      const isValidPrompt = nextNumTokensEstimate <= maxNumTokens

      if (prompt && !isValidPrompt) {
        break
      }

      messages = nextMessages
      numTokens = nextNumTokensEstimate

      if (!isValidPrompt) {
        break
      }

      if (!parentMessageId) {
        break
      }

      const parentMessage = await this._getMessageById(parentMessageId)
      if (!parentMessage) {
        break
      }

      const parentMessageRole = parentMessage.role || 'user'

      nextMessages = nextMessages.slice(0, systemMessageOffset).concat([
        {
          role: parentMessageRole as any,
          content: parentMessage.text,
          name: parentMessage.name
        },
        ...nextMessages.slice(systemMessageOffset)
      ])

      parentMessageId = parentMessage.parentMessageId
    } while (true)

    // Use up to 4096 tokens (prompt + response), but try to leave 1000 tokens
    // for the response.
    const maxTokens = Math.max(
      1,
      Math.min(this._maxModelTokens - numTokens, this._maxResponseTokens)
    )

    return { messages, maxTokens, numTokens }
  }

  protected async _getTokenCount(text: string) {
    // TODO: use a better fix in the tokenizer
    text = text.replace(/<\|endoftext\|>/g, '')

    return tokenizer.encode(text).length
  }

  protected async _defaultGetMessageById(
    id: string
  ): Promise<types.ChatMessage> {
    const res = await this._messageStore.get(id)
    return res
  }

  protected async _defaultUpsertMessage(
    message: types.ChatMessage
  ): Promise<void> {
    await this._messageStore.set(message.id, message)
  }
}
