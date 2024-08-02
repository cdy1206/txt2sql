import chainlit as cl
from chainlit.input_widget import Select
import os

from dotenv import load_dotenv
from dquestion.deepseek.deepseek_coder import DeepseekCoder

load_dotenv()
# dq = DeepseekCoder(config={"api_key": os.environ.get("API_KEY")})
dq = DeepseekCoder()
dq.connect_to_sqlite('../data/Chinook.sqlite')


@cl.step(root=True, language="sql", name="DQuestion")
async def gen_query(human_query: str):
    sql_query = dq.generate_sql(human_query)
    return sql_query


@cl.step(root=True, name="DQuestion")
async def execute_query(query):
    current_step = cl.context.current_step
    df = dq.run_sql(query)
    current_step.output = df.head().to_markdown(index=False)

    return df


@cl.step(name="Plot", language="python")
async def plot(human_query, sql, df):
    current_step = cl.context.current_step
    plotly_code = dq.generate_plotly_code(question=human_query, sql=sql, df=df)
    fig = dq.get_plotly_figure(plotly_code=plotly_code, df=df)

    current_step.output = plotly_code
    return fig


@cl.step(type="run", root=True, name="DQuestion")
async def chain(human_query: str):
    sql_query = await gen_query(human_query)
    df = await execute_query(sql_query)
    fig = await plot(human_query, sql_query, df)

    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
    await cl.Message(content=human_query, elements=elements, author="DQuestion").send()


@cl.on_message
async def main(message: cl.Message):
    await chain(message.content)


@cl.on_chat_start
async def setup():
    await cl.Avatar(
        name="Dquestion",
        url="assets/logo.png",
    ).send()

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Qwen-Long",
                values=["qwen-long"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Model"]
