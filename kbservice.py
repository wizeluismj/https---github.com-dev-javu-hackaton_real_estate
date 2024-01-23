import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
import os
from flask import Flask, request

app = Flask(__name__)
df = pd.read_csv("realtor-data.csv")
os.environ["OPENAI_API_KEY"] = "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"
agent = create_pandas_dataframe_agent(OpenAI(temperature=0.1), df, verbose=True)

@app.route('/question', methods=['POST'])
def answer_question():
    question = request.form['question']
    # answer = agent.invoke(question + " in JSON format")
    answer = agent.invoke(question)
    return answer

if __name__ == '__main__':
    app.run(debug=True)