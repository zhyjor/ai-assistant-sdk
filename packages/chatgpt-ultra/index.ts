import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'https://www.jiujiuai.life/v1',
  apiKey: 'sk-YqJXon9AiYv679yXCa6487De783f475cA99366806e65D2Ab', // This is the default and can be omitted
});

async function main() {
  const stream = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: 'Say this is a test' }],
    stream: false,
  });
  console.log(JSON.stringify(stream.choices[0]));
  // for await (const chunk of stream) {
  //   console.log(JSON.stringify(chunk.choices[0]));
  //   // console.log(chunk.choices[0]?.delta?.content);
  //   process.stdout.write(chunk.choices[0]?.delta?.content || '');
  // }
}
main();
