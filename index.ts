import readline from 'node:readline/promises'
import { StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import { MemorySaver } from '@langchain/langgraph';
import { ChatGroq } from '@langchain/groq';
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";

const checkpointer = new MemorySaver();

const tool = new TavilySearch({
  maxResults: 5,
  topic: "general",
});

const tools: any[] = [tool];
const toolNode = new ToolNode(tools);

const llm = new ChatGroq({
    model: "openai/gpt-oss-120b",
    temperature: 0,
    maxRetries: 2, 
}).bindTools(tools);

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// 1. Define node function
async function callModel(state) {
    console.log('Calling LLM....');
    try {
        const response = await llm.invoke(state.messages);
        return { messages: [response] };
    } catch (err: any) {
        // ✅ handle Groq rate limit error
        if (err.status === 429) {
            const retryAfter = parseInt(err?.headers?.["retry-after"] || "30", 10) * 1000;
            console.warn(`⚠️ Rate limit hit. Retrying in ${retryAfter / 1000}s...`);
            await sleep(retryAfter);
            return await callModel(state); // retry once after waiting
        }
        throw err; // rethrow other errors
    }
}

function shouldContinue(state){
    const lastMessage = state.messages[state.messages.length - 1];
    if(lastMessage.tool_calls?.length > 0){
        return 'tools';
    }
    return '__end__';
}

// 2. Build the graph
const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addNode('tools', toolNode)
    .addEdge('__start__', 'agent')
    .addEdge('tools', 'agent')
    .addConditionalEdges('agent', shouldContinue);

const app = workflow.compile({checkpointer});

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    })
    while (true) {
        const userInput = await rl.question('You: ');
        if (userInput == 'bye') break;

        const finalState = await app.invoke({
            messages: [{ role: 'user', content: userInput }],
        }, {configurable: {thread_id: '1'}});

        const lastMessage = finalState.messages[finalState.messages.length - 1];
        console.log('AI:', lastMessage?.content);
    }
    rl.close();
}

main();
