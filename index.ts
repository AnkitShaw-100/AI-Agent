import readline from 'node:readline/promises'
import { StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatGroq } from '@langchain/groq';

// 1. Define node functions
// 2. Build the graph
// 3. Compile and invoke the graph


const llm = new ChatGroq({
    model: "openai/gpt-oss-120b",
    temperature: 0,
    maxRetries: 2,
});

// 1. Define node function
async function callModel(state) {
    // Calling the LLM using APIs
    console.log('Calling LLM....');

    const response = await llm.invoke(state.messages);
    return { messages: [response] };
}

// 2. Build the graph
const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', callModel)
    .addEdge('__start__', 'agent')
    .addEdge('agent', '__end__');


const app = workflow.compile();

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
        });

        const lastMessage = finalState.messages[finalState.messages.length - 1];
        console.log('AI:', lastMessage?.content);
    }
    rl.close();
}

main();