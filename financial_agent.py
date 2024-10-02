from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()
    return os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(
    model = "gpt-4-turbo",
    api_key=configure(),
)

prompt = '''
    Como assistente financeiro pessoal, que responderá as perguntas dando dicas financeiras e de investimentos.
    Responda tudo em português brasileiro.
    Perguntas: {query}
'''
prompt_template = PromptTemplate.from_template(prompt) 

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Um shell Python. Use isso para executar código Python. Execute apenas códigos Python válidos.'
                'Se você precisar obter o retorno do código, use a função "print(...)".'
                'Use para realizar cálculos financeiros necessários para responder as perguntas e dar dicas.',
    func=python_repl.run
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name='Busca DuckDuckGo',
    description='Útil para encontrar informações e dicas de economia e opções de investimento.'
                'Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta, não'
                'responda diretamente. Sua resposta deve informar que há elementos pesquisados na internet.',
    func=search.run,
)

react_instructions = hub.pull('hwchase17/react')

tools = [python_repl_tool, search_tool]

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_instructions,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

question = '''
    Minha renda é de R$10.000 por mês, o total das minhas despesas é de R$12.500 mais R$1.000 de aluguel.
    Quais dicas de investimento você me daria para o restante do dinheiro?
'''

output = agent_executor.invoke(
    {'input': prompt_template.format(query=question)}
)

print(output.get('output'))