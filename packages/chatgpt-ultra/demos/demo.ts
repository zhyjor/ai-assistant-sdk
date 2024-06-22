import dotenv from 'dotenv-safe'
import { oraPromise } from 'ora'

import { ChatGPTAPI } from '../src'

dotenv.config()

/**
 * Demo CLI for testing basic functionality.
 *
 * ```
 * npx tsx demos/demo.ts
 * ```
 */
async function main() {
  const api = new ChatGPTAPI({
    baseURL: 'https://www.jiujiuai.life/v1',
    apiKey: '', // This is the d
    debug: false
  })

  const prompt =
    'Write a python version of bubble sort. Do not include example usage.'

  const res = await oraPromise(api.sendMessage(prompt), {
    text: prompt
  })
  console.log(res)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
